# ComfyUI-AsymFlow

<p align="center">
  <img src="assets/hero.png" alt="AsymFLUX.2 Klein sample outputs" />
</p>

Standalone ComfyUI node pack for [AsymFLUX.2 Klein](https://hanshengchen.com/asymflow) -- pixel-space text-to-image generation.

Based on [AsymFlow: Asymmetric Flow Models](https://arxiv.org/abs/2605.12964) by Hansheng Chen et al. (Stanford University).
Core inference code extracted from [LakonLab](https://github.com/Lakonik/LakonLab).

## Nodes

| Node | Purpose |
|------|---------|
| **AsymFLUX.2 Klein Loader** | Load transformer + text encoder + adapter |
| **AsymFLUX.2 Klein Sampler** | Text-to-image and image-to-image generation |

## Setup

### 1. install custom nodes.

```bash
git clone https://github.com/RealRebelAI/REBELS_LOW_VRAM_ComfyUI-Klein9B-AsymFlow.git

cd ComfyUI/custom_nodes/REBELS_LOW_VRAM_ComfyUI-Klein9B-AsymFlow
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

### 2. Install dependencies

```bash
cd ComfyUI/custom_nodes/REBELS_LOW_VRAM_ComfyUI-Klein9B-AsymFlow
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt

OR

cd ComfyUI/custom_nodes/REBELS_LOW_VRAM_ComfyUI-Klein9B-AsymFlow
pip install -r requirements.txt
```

### 3. Download Model files

- bf16 model - /models/diffusion_models/

https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/flux-2-klein-9b.safetensors



- gguf model - /models/unet/

https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/tree/main


- fp4 encoder (smallest) - /models/text_encoders/

https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b_fp4mixed.safetensors


- fp8 encoder (better but larger) - /models/text_encoders/

https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors



REQUIRED Adapter LoRA (rename to "ASYM_FLUX_KLEIN_9B") - /models/loras/

https://huggingface.co/Lakonik/AsymFLUX.2-klein-9B/blob/main/diffusion_pytorch_model.safetensors


### 4. Use in ComfyUI

workflow:
https://civitai.com/models/2626000/rebels-asym-flux-2-klein-9b?modelVersionId=2948288

## Model Locations

| Component | Folder | Format |
|-----------|--------|--------|
| Transformer | `models/diffusion_models/` | Single .safetensors |
| Text Encoder | `models/text_encoders/<name>/` | Directory (config.json + sharded .safetensors) |
| Adapter | `models/loras/` | Single .safetensors |

The text encoder directory supports both flat layout and BFL-style subdirectories (`text_encoder/` + `tokenizer/`).

## Recommended Settings

| Parameter | Default | Notes |
|-----------|---------|-------|
| Steps | 38 | |
| Guidance Scale | 4.0 | |
| Orthogonal Guidance | 1.0 | Controls CFG orthogonality |
| Clamp Denoised | True | Improves color accuracy |
| dtype | bfloat16 | Use float16 if bfloat16 unsupported |

## Credits

- **AsymFlow**: Hansheng Chen, Jan Ackermann, Minseo Kim, Gordon Wetzstein, Leonidas Guibas (Stanford University)
- **FLUX.2 Klein**: Black Forest Labs
- **Rebel AI**: Custom Node Fork Contributor For GGUF and CLIP

## License

The bundled inference code in `asymflow_lib/` is derived from [LakonLab](https://github.com/Lakonik/LakonLab). Please refer to the original repository for licensing terms.
