"""
Rebels Pixel Smooth -- Anti-Tiling and Patch Coherence Modifier for ComfyUI.
Intercepts transformer attention layers natively using standard ComfyUI MODEL routing.
"""

import math
import logging
import torch

logger = logging.getLogger("[RebelsPixelSmooth]")

class RebelsPixelSmooth:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # MATCHED: Now accepts a standard ComfyUI MODEL wire connection
                "model": ("MODEL",),
                "smoothing_strength": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05}),
                "patch_size": ("INT", {"default": 16, "min": 8, "max": 64, "step": 8}),
            }
        }

    # MATCHED: Returns a standard ComfyUI MODEL wire connection
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_smoothing"
    CATEGORY = "AsymFlow/Modifiers"

    def apply_smoothing(self, model, smoothing_strength, patch_size):
        if smoothing_strength == 0.0:
            return (model,)

        # Create a shallow clone of the model container to prevent corrupting core RAM caches
        patched_model = model.clone()
        transformer = patched_model.model.diffusion_model

        # Hook into the double_blocks (joint attention) layers inside the base transformer
        for name, block in transformer.named_modules():
            if "transformer_blocks" in name and hasattr(block, "attn"):
                orig_forward = block.attn.forward
                
                def patched_attn_forward(*args, **kwargs):
                    hidden_states = orig_forward(*args, **kwargs)
                    
                    # hidden_states shape: [batch, sequence_len, hidden_dim]
                    b, s, d = hidden_states.shape
                    spatial_side = int(math.sqrt(s))
                    
                    if spatial_side * spatial_side == s:
                        # Reshape sequence back to a 4D spatial feature map: [B, D, H, W]
                        x = hidden_states.transpose(1, 2).view(b, d, spatial_side, spatial_side)
                        
                        # Apply localized neighborhood neighborhood blur to the patch boundaries
                        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=x.dtype, device=x.device)
                        kernel = kernel / kernel.sum()
                        kernel = kernel.view(1, 1, 3, 3).repeat(d, 1, 1, 1)
                        
                        padded_x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='replicate')
                        smoothed_x = torch.nn.functional.conv2d(padded_x, kernel, groups=d)
                        
                        # Linearly blend based on custom strength parameter
                        blended_x = (1.0 - smoothing_strength) * x + smoothing_strength * smoothed_x
                        hidden_states = blended_x.view(b, d, s).transpose(1, 2)
                        
                    return hidden_states

                block.attn.forward = patched_attn_forward

        logger.info(f"[Rebels AI] Flawlessly hooked transformer models with custom pixel smoothing at strength: {smoothing_strength}")
        return (patched_model,)

NODE_CLASS_MAPPINGS = {
    "RebelsPixelSmooth": RebelsPixelSmooth
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelsPixelSmooth": "Rebels Pixel Smooth"
}
