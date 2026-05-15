from .nodes import (
    AsymFlux2KleinLoader,
    AsymFlux2KleinSampler,
    AsymFlux2KleinLoaderNoCLIP,
    AsymFlux2KleinCondSampler,
    AsymFlux2KleinLoaderGGUF
)

NODE_CLASS_MAPPINGS = {
    "AsymFlux2KleinLoader": AsymFlux2KleinLoader,
    "AsymFlux2KleinSampler": AsymFlux2KleinSampler,
    "AsymFlux2KleinLoaderNoCLIP": AsymFlux2KleinLoaderNoCLIP,
    "AsymFlux2KleinCondSampler": AsymFlux2KleinCondSampler,
    "AsymFlux2KleinLoaderGGUF": AsymFlux2KleinLoaderGGUF,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsymFlux2KleinLoader": "AsymFLUX.2 Klein Loader",
    "AsymFlux2KleinSampler": "AsymFLUX.2 Klein Sampler",
    "AsymFlux2KleinLoaderNoCLIP": "AsymFLUX.2 Klein Loader (No CLIP)",
    "AsymFlux2KleinCondSampler": "AsymFLUX.2 Klein Cond Sampler",
    "AsymFlux2KleinLoaderGGUF": "AsymFLUX.2 Klein Loader (GGUF)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
