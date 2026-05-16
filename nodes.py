"""
AsymFLUX.2 Klein -- Pixel-space text-to-image generation for ComfyUI.
Based on AsymFlow by Hansheng Chen et al. (Stanford University)
"""

import os
import math
import logging
import torch
import numpy as np
import folder_paths
from safetensors.torch import load_file

logger = logging.getLogger("[AsymFlow]")

_pipe_cache = {}

_ASYMFLUX2_KLEIN_CONFIG = {
    "patch_size": 1,
    "in_channels": 128,
    "hidden_size": 4096,
    "base_rank": 128,
    "num_layers": 8,
    "num_single_layers": 24,
    "attention_head_dim": 128,
    "num_attention_heads": 32,
    "joint_attention_dim": 12288,
    "timestep_guidance_channels": 256,
    "mlp_ratio": 3.0,
    "axes_dims_rope": (32, 32, 32, 32),
    "rope_theta": 2000,
    "eps": 1e-6,
    "sigma_min": 1e-4,
    "num_timesteps": 1,
    "guidance_embeds": False,
}

def _get_dtype(name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]

def _list_model_dirs(folder_name):
    dirs = []
    try:
        paths = folder_paths.get_folder_paths(folder_name)
    except KeyError:
        return dirs
    for base_path in paths:
        if not os.path.isdir(base_path):
            continue
        for d in sorted(os.listdir(base_path)):
            if os.path.isdir(os.path.join(base_path, d)):
                dirs.append(d)
    return dirs

def _resolve_model_dir(folder_name, dir_name):
    try:
        paths = folder_paths.get_folder_paths(folder_name)
    except KeyError:
        paths = []
    for base_path in paths:
        full = os.path.join(base_path, dir_name)
        if os.path.isdir(full):
            return full
    raise FileNotFoundError(f"Directory '{dir_name}' not found in {folder_name}/ model folder.")

def _cleanup_meta(model, dtype):
    """
    Cleans up stranded meta tensors AFTER base weights are assigned,
    but BEFORE LoRA is loaded.
    """
    for module in model.modules():
        for name, param in module.named_parameters(recurse=False):
            if param is not None and param.device.type == "meta":
                new_tensor = torch.zeros_like(param, device="cpu", dtype=dtype)
                module.register_parameter(name, torch.nn.Parameter(new_tensor, requires_grad=param.requires_grad))
        for name, buffer in module.named_buffers(recurse=False):
            if buffer is not None and buffer.device.type == "meta":
                new_buffer = torch.zeros_like(buffer, device="cpu", dtype=dtype)
                module.register_buffer(name, new_buffer)

def _convert_comfy_to_diffusers_format(comfy_sd, dtype):
    """
    Translates ComfyUI keys to Diffusers format. Uses .clone() on chunks 
    to prevent memory fragmentation without the massive overhead of contiguous().
    """
    logger.info("[AsymFlow] Translating ComfyUI keys to Diffusers format...")
    diffusers_sd = {}

    for k, v in comfy_sd.items():
        k_src = k.removeprefix("transformer.") if k.startswith("transformer.") else k

        if k_src.startswith("time_in."):
            diffusers_sd[k_src.replace("time_in.in_layer", "time_text_embed.timestep_embedder.linear_1")] = v
            continue
        if k_src.startswith("vector_in."):
            diffusers_sd[k_src.replace("vector_in.in_layer", "time_text_embed.text_embedder.linear_1")] = v
            continue
        if k_src.startswith("guidance_in."):
            diffusers_sd[k_src.replace("guidance_in.in_layer", "time_guidance_embed.timestep_embedder.linear_1")] = v
            continue

        if "double_blocks." in k_src:
            block_idx = k_src.split(".")[1]
            suffix = k_src.split(f"double_blocks.{block_idx}.")[1]
            diff_prefix = f"transformer_blocks.{block_idx}."

            if suffix.startswith("img_attn.qkv."):
                chunks = torch.chunk(v, 3, dim=0)
                diffusers_sd[f"{diff_prefix}attn.to_q.weight" if "weight" in suffix else f"{diff_prefix}attn.to_q.bias"] = chunks[0].clone()
                diffusers_sd[f"{diff_prefix}attn.to_k.weight" if "weight" in suffix else f"{diff_prefix}attn.to_k.bias"] = chunks[1].clone()
                diffusers_sd[f"{diff_prefix}attn.to_v.weight" if "weight" in suffix else f"{diff_prefix}attn.to_v.bias"] = chunks[2].clone()
                continue
            if suffix.startswith("txt_attn.qkv."):
                chunks = torch.chunk(v, 3, dim=0)
                diffusers_sd[f"{diff_prefix}attn_context.to_q.weight" if "weight" in suffix else f"{diff_prefix}attn_context.to_q.bias"] = chunks[0].clone()
                diffusers_sd[f"{diff_prefix}attn_context.to_k.weight" if "weight" in suffix else f"{diff_prefix}attn_context.to_k.bias"] = chunks[1].clone()
                diffusers_sd[f"{diff_prefix}attn_context.to_v.weight" if "weight" in suffix else f"{diff_prefix}attn_context.to_v.bias"] = chunks[2].clone()
                continue
            if suffix.startswith("img_attn.proj."):
                diffusers_sd[f"{diff_prefix}attn.to_out.0.weight" if "weight" in suffix else f"{diff_prefix}attn.to_out.0.bias"] = v
                continue
            if suffix.startswith("txt_attn.proj."):
                diffusers_sd[f"{diff_prefix}attn_context.to_out.weight" if "weight" in suffix else f"{diff_prefix}attn_context.to_out.bias"] = v
                continue
            if suffix.startswith("img_mlp."):
                new_s = suffix.replace("img_mlp.0", "ff.linear_in").replace("img_mlp.2", "ff.linear_out")
                diffusers_sd[f"{diff_prefix}{new_s}"] = v
                continue
            if suffix.startswith("txt_mlp."):
                new_s = suffix.replace("txt_mlp.0", "ff_context.linear_in").replace("txt_mlp.2", "ff_context.linear_out")
                diffusers_sd[f"{diff_prefix}{new_s}"] = v
                continue
            if suffix.startswith("img_mod.lin."):
                diffusers_sd[f"{diff_prefix}norm1.linear.weight" if "weight" in suffix else f"{diff_prefix}norm1.linear.bias"] = v
                continue
            if suffix.startswith("txt_mod.lin."):
                diffusers_sd[f"{diff_prefix}norm1_context.linear.weight" if "weight" in suffix else f"{diff_prefix}norm1_context.linear.bias"] = v
                continue

        if "single_blocks." in k_src:
            block_idx = k_src.split(".")[1]
            suffix = k_src.split(f"single_blocks.{block_idx}.")[1]
            diff_prefix = f"single_transformer_blocks.{block_idx}."

            if suffix.startswith("linear1."):
                chunks = torch.chunk(v, 4, dim=0)
                diffusers_sd[f"{diff_prefix}attn.to_q.weight" if "weight" in suffix else f"{diff_prefix}attn.to_q.bias"] = chunks[0].clone()
                diffusers_sd[f"{diff_prefix}attn.to_k.weight" if "weight" in suffix else f"{diff_prefix}attn.to_k.bias"] = chunks[1].clone()
                diffusers_sd[f"{diff_prefix}attn.to_v.weight" if "weight" in suffix else f"{diff_prefix}attn.to_v.bias"] = chunks[2].clone()
                diffusers_sd[f"{diff_prefix}ff.linear_in.weight" if "weight" in suffix else f"{diff_prefix}ff.linear_in.bias"] = chunks[3].clone()
                continue
            if suffix.startswith("linear2."):
                diffusers_sd[f"{diff_prefix}attn.to_out.weight" if "weight" in suffix else f"{diff_prefix}attn.to_out.bias"] = v
                continue
            if suffix.startswith("modulation.lin."):
                diffusers_sd[f"{diff_prefix}norm1.linear.weight" if "weight" in suffix else f"{diff_prefix}norm1.linear.bias"] = v
                continue

        if k_src.startswith("img_in."):
            diffusers_sd[k_src.replace("img_in", "x_embedder")] = v
            continue
        if k_src.startswith("txt_in."):
            diffusers_sd[k_src.replace("txt_in", "context_embedder")] = v
            continue
        if k_src.startswith("final_layer.linear."):
            diffusers_sd[k_src.replace("final_layer.linear", "proj_out")] = v
            continue
        if k_src.startswith("final_layer.adaLN_modulation.lin."):
            diffusers_sd[k_src.replace("final_layer.adaLN_modulation.lin", "norm_out.linear")] = v
            continue

        diffusers_sd[k_src] = v

    return diffusers_sd

class AsymFlux2KleinLoader:
    @classmethod
    def INPUT_TYPES(cls):
        diff_models = folder_paths.get_filename_list("diffusion_models")
        te_dirs = sorted(set(_list_model_dirs("text_encoders") + _list_model_dirs("clip")))
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "transformer": (diff_models, {}),
                "text_encoder": (te_dirs if te_dirs else ["(place model dir in text_encoders/ or clip/)"], {}),
                "adapter": (loras, {}),
            },
            "optional": {
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "enable_cpu_offload": ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
            },
        }

    RETURN_TYPES = ("ASYMFLUX_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "AsymFlow"

    def load(self, transformer, text_encoder, adapter, dtype="bfloat16", device="cuda", enable_cpu_offload=True):
        transformer_path = folder_paths.get_full_path("diffusion_models", transformer)
        adapter_path = folder_paths.get_full_path("loras", adapter) 
        try:
            te_dir = _resolve_model_dir("text_encoders", text_encoder)
        except FileNotFoundError:
            te_dir = _resolve_model_dir("clip", text_encoder)

        cache_key = (transformer_path, te_dir, adapter_path, dtype, device, enable_cpu_offload)
        if cache_key in _pipe_cache:
            return (_pipe_cache[cache_key],)

        torch_dtype = _get_dtype(dtype)

        from accelerate import init_empty_weights
        from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
        from .asymflow_lib import PixelFlux2KleinPipeline, OklabColorEncoder, FlowAdapterScheduler
        from .asymflow_lib.asymflux2_model import AsymFlux2Transformer2DModel

        raw_state_dict = load_file(transformer_path, device="cpu")
        base_state_dict = _convert_comfy_to_diffusers_format(raw_state_dict, torch_dtype)

        with init_empty_weights():
            transformer_model = AsymFlux2Transformer2DModel(**_ASYMFLUX2_KLEIN_CONFIG)

        transformer_model.load_state_dict(base_state_dict, strict=False, assign=True)

        # 1. Clean Meta First
        _cleanup_meta(transformer_model, torch_dtype)

        # 2. Load LoRA Second
        adapter_state_dict = load_file(adapter_path, device="cpu")
        lora_state_dict = {k.removeprefix("transformer."): v.to(dtype=torch_dtype) for k, v in adapter_state_dict.items() if "lora" in k}
        
        if lora_state_dict:
            transformer_model.load_lora_adapter(lora_state_dict, prefix=None, adapter_name="asymflow", low_cpu_mem_usage=True)

        te_subdir = os.path.join(te_dir, "text_encoder")
        tok_subdir = os.path.join(te_dir, "tokenizer")
        te_load_path = te_subdir if os.path.isdir(te_subdir) else te_dir
        tok_load_path = tok_subdir if os.path.isdir(tok_subdir) else te_dir

        text_encoder_model = Qwen3ForCausalLM.from_pretrained(te_load_path, torch_dtype=torch_dtype, local_files_only=True)
        tokenizer = Qwen2TokenizerFast.from_pretrained(tok_load_path, local_files_only=True)

        pipe = PixelFlux2KleinPipeline(
            transformer=transformer_model,
            text_encoder=text_encoder_model,
            tokenizer=tokenizer,
            vae=OklabColorEncoder(use_affine_norm=True, mean=(0.56, 0.0, 0.01), std=0.16),
            scheduler=FlowAdapterScheduler(shift=17.0, use_dynamic_shifting=True, base_seq_len=1024**2, base_scheduler="UniPCMultistep")
        )

        pipe = pipe.to(dtype=torch_dtype)
        
        if enable_cpu_offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe = pipe.to(device)

        _pipe_cache[cache_key] = pipe
        return (pipe,)

class AsymFlux2KleinSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ASYMFLUX_PIPE",),
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "Low quality, blurry"}),
                "width": ("INT", {"default": 1024, "step": 16}),
                "height": ("INT", {"default": 1024, "step": 16}),
                "num_inference_steps": ("INT", {"default": 38}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "step": 0.1}),
                "orthogonal_guidance": ("FLOAT", {"default": 1.0, "step": 0.1}),
                "clamp_denoised": ("BOOLEAN", {"default": True}),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "AsymFlow"

    def generate(self, pipe, prompt, seed, negative_prompt="Low quality, blurry", width=1024, height=1024, num_inference_steps=38, guidance_scale=4.0, orthogonal_guidance=1.0, clamp_denoised=True, image=None):
        from PIL import Image
        input_image = None
        if image is not None:
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            input_image = Image.fromarray(img_np)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            orthogonal_guidance=orthogonal_guidance,
            clamp_denoised=clamp_denoised,
            generator=generator
        )

        img_tensor = torch.from_numpy(np.array(result.images[0]).astype(np.float32) / 255.0).unsqueeze(0)
        return (img_tensor,)

class AsymFlux2KleinLoaderNoCLIP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transformer": (folder_paths.get_filename_list("diffusion_models"), {}),
                "adapter": (folder_paths.get_filename_list("loras"), {}),
            },
            "optional": {
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("ASYMFLUX_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "AsymFlow"

    def load(self, transformer, adapter, dtype="bfloat16", device="cuda"):
        transformer_path = folder_paths.get_full_path("diffusion_models", transformer)
        adapter_path = folder_paths.get_full_path("loras", adapter)

        cache_key = ("no_clip", transformer_path, adapter_path, dtype, device)
        if cache_key in _pipe_cache:
            return (_pipe_cache[cache_key],)

        torch_dtype = _get_dtype(dtype)

        from accelerate import init_empty_weights
        from .asymflow_lib import PixelFlux2KleinPipeline, OklabColorEncoder, FlowAdapterScheduler
        from .asymflow_lib.asymflux2_model import AsymFlux2Transformer2DModel

        raw_state_dict = load_file(transformer_path, device="cpu")
        base_state_dict = _convert_comfy_to_diffusers_format(raw_state_dict, torch_dtype)

        with init_empty_weights():
            transformer_model = AsymFlux2Transformer2DModel(**_ASYMFLUX2_KLEIN_CONFIG)

        transformer_model.load_state_dict(base_state_dict, strict=False, assign=True)

        # 1. Clean Meta First
        _cleanup_meta(transformer_model, torch_dtype)

        # 2. Load LoRA Second
        adapter_state_dict = load_file(adapter_path, device="cpu")
        lora_state_dict = {k.removeprefix("transformer."): v.to(dtype=torch_dtype) for k, v in adapter_state_dict.items() if "lora" in k}

        if lora_state_dict:
            transformer_model.load_lora_adapter(lora_state_dict, prefix=None, adapter_name="asymflow", low_cpu_mem_usage=True)

        pipe = PixelFlux2KleinPipeline(
            transformer=transformer_model,
            text_encoder=None,
            tokenizer=None,
            vae=OklabColorEncoder(use_affine_norm=True, mean=(0.56, 0.0, 0.01), std=0.16),
            scheduler=FlowAdapterScheduler(shift=17.0, use_dynamic_shifting=True, base_seq_len=1024**2, base_scheduler="UniPCMultistep")
        )

        pipe = pipe.to(dtype=torch_dtype)
        pipe.enable_sequential_cpu_offload()

        _pipe_cache[cache_key] = pipe
        return (pipe,)

class AsymFlux2KleinLoaderGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",), 
                "adapter": (folder_paths.get_filename_list("loras"), {}),
            },
            "optional": {
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("ASYMFLUX_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load"
    CATEGORY = "AsymFlow"

    def load(self, model, adapter, dtype="bfloat16", device="cuda"):
        adapter_path = folder_paths.get_full_path("loras", adapter)
        torch_dtype = _get_dtype(dtype)

        from accelerate import init_empty_weights
        from .asymflow_lib import PixelFlux2KleinPipeline, OklabColorEncoder, FlowAdapterScheduler
        from .asymflow_lib.asymflux2_model import AsymFlux2Transformer2DModel

        source_model = model.model.diffusion_model
        raw_state_dict = {k: v.to(torch_dtype) for k, v in source_model.state_dict().items()}

        base_state_dict = _convert_comfy_to_diffusers_format(raw_state_dict, torch_dtype)

        with init_empty_weights():
            transformer_model = AsymFlux2Transformer2DModel(**_ASYMFLUX2_KLEIN_CONFIG)

        transformer_model.load_state_dict(base_state_dict, strict=False, assign=True)

        # 1. Clean Meta First
        _cleanup_meta(transformer_model, torch_dtype)

        # 2. Load LoRA Second
        adapter_state_dict = load_file(adapter_path, device="cpu")
        lora_state_dict = {k.removeprefix("transformer."): v.to(dtype=torch_dtype) for k, v in adapter_state_dict.items() if "lora" in k}

        if lora_state_dict:
            transformer_model.load_lora_adapter(lora_state_dict, prefix=None, adapter_name="asymflow", low_cpu_mem_usage=True)

        pipe = PixelFlux2KleinPipeline(
            transformer=transformer_model,
            text_encoder=None,
            tokenizer=None,
            vae=OklabColorEncoder(use_affine_norm=True, mean=(0.56, 0.0, 0.01), std=0.16),
            scheduler=FlowAdapterScheduler(shift=17.0, use_dynamic_shifting=True, base_seq_len=1024**2, base_scheduler="UniPCMultistep")
        )

        pipe = pipe.to(dtype=torch_dtype)
        pipe.enable_sequential_cpu_offload()

        return (pipe,)

class AsymFlux2KleinCondSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ASYMFLUX_PIPE",),
                "positive": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "width": ("INT", {"default": 1024, "step": 16}),
                "height": ("INT", {"default": 1024, "step": 16}),
                "num_inference_steps": ("INT", {"default": 38}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "step": 0.1}),
                "orthogonal_guidance": ("FLOAT", {"default": 1.0, "step": 0.1}),
                "clamp_denoised": ("BOOLEAN", {"default": True}),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "AsymFlow"

    def generate(self, pipe, positive, seed, negative=None, width=1024, height=1024, num_inference_steps=38, guidance_scale=4.0, orthogonal_guidance=1.0, clamp_denoised=True, image=None):
        from PIL import Image

        target_dtype = pipe.transformer.dtype
        target_device = pipe._execution_device

        p_embeds = positive[0][0].to(target_device).to(target_dtype)
        p_embeds = torch.nan_to_num(p_embeds, nan=0.0, posinf=10.0, neginf=-10.0)

        if negative is not None:
            n_embeds = negative[0][0].to(target_device).to(target_dtype)
            n_embeds = torch.nan_to_num(n_embeds, nan=0.0, posinf=10.0, neginf=-10.0)
        else:
            n_embeds = torch.zeros_like(p_embeds)

        input_image = None
        if image is not None:
            img_np = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            input_image = Image.fromarray(img_np)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        result = pipe(
            prompt=None,
            prompt_embeds=p_embeds,
            negative_prompt=None,
            negative_prompt_embeds=n_embeds,
            image=input_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            orthogonal_guidance=orthogonal_guidance,
            clamp_denoised=clamp_denoised,
            generator=generator
        )

        img_tensor = torch.from_numpy(np.array(result.images[0]).astype(np.float32) / 255.0).unsqueeze(0)
        return (img_tensor,)
