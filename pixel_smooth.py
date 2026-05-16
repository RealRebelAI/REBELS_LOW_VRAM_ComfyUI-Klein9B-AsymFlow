"""
AsymFLUX.2 Klein -- Patch Smoothing and Anti-Tiling Layer for ComfyUI.
Intercepts joint attention blocks to blend patch-boundary frequencies.
"""

import logging
import torch

logger = logging.getLogger("[AsymFlowSmooth]")

class AsymFluxPatchSmoother:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("ASYMFLUX_PIPE",),
                "smoothing_strength": ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0, "step": 0.05}),
                "patch_size": ("INT", {"default": 16, "min": 8, "max": 64, "step": 8}),
            }
        }

    RETURN_TYPES = ("ASYMFLUX_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "apply_smoothing"
    CATEGORY = "AsymFlow/Modifiers"

    def apply_smoothing(self, pipe, smoothing_strength, patch_size):
        if smoothing_strength == 0.0:
            return (pipe,)

        # Shallow copy the pipeline pipeline container to keep caching clean
        smoothed_pipe = pipe
        transformer = smoothed_pipe.transformer

        # Hook into the double_blocks (joint attention) where spatial grids are processed
        for name, block in transformer.named_modules():
            if "transformer_blocks" in name and hasattr(block, "attn"):
                # Inject structural neighborhood blending into the attention forward pass
                orig_forward = block.attn.forward
                
                def patched_attn_forward(*args, **kwargs):
                    hidden_states = orig_forward(*args, **kwargs)
                    
                    # hidden_states shape: [batch, sequence_len, hidden_dim]
                    # Map sequence back to 2D spatial grid to smooth adjacent boundaries
                    b, s, d = hidden_states.shape
                    spatial_side = int(math.sqrt(s))
                    
                    if spatial_side * spatial_side == s:
                        # Reshape to 4D feature map: [B, D, H, W]
                        x = hidden_states.transpose(1, 2).view(b, d, spatial_side, spatial_side)
                        
                        # Apply a localized Gaussian-like neighborhood blur to the feature boundaries
                        kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=x.dtype, device=x.device)
                        kernel = kernel / kernel.sum()
                        kernel = kernel.view(1, 1, 3, 3).repeat(d, 1, 1, 1)
                        
                        # Pad boundaries to keep dimensions identical
                        padded_x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='replicate')
                        smoothed_x = torch.nn.functional.conv2d(padded_x, kernel, groups=d)
                        
                        # Linearly blend based on user strength parameter
                        blended_x = (1.0 - smoothing_strength) * x + smoothing_strength * smoothed_x
                        hidden_states = blended_x.view(b, d, s).transpose(1, 2)
                        
                    return hidden_states

                block.attn.forward = patched_attn_forward

        logger.info(f"[AsymFlow] Successfully hooked joint attention grids with smoothing strength: {smoothing_strength}")
        return (smoothed_pipe,)

NODE_CLASS_MAPPINGS = {
    "AsymFluxPatchSmoother": AsymFluxPatchSmoother
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsymFluxPatchSmoother": "AsymFLUX.2 Patch Smoother (Anti-Tiling)"
}