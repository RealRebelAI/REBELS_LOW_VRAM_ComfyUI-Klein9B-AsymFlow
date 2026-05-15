from .nodes import AsymFlux2KleinLoader, AsymFlux2KleinSampler
from .nodes import AsymFlux2KleinLoaderNoCLIP, AsymFlux2KleinCondSampler

NODE_CLASS_MAPPINGS = {
    "AsymFlux2KleinLoader": AsymFlux2KleinLoader,
    "AsymFlux2KleinSampler": AsymFlux2KleinSampler,
    "AsymFlux2KleinLoaderNoCLIP": AsymFlux2KleinLoaderNoCLIP,
    "AsymFlux2KleinCondSampler": AsymFlux2KleinCondSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsymFlux2KleinLoader": "AsymFLUX.2 Klein Loader",
    "AsymFlux2KleinSampler": "AsymFLUX.2 Klein Sampler",
    "AsymFlux2KleinLoaderNoCLIP": "AsymFLUX.2 Klein Loader (No CLIP)",
    "AsymFlux2KleinCondSampler": "AsymFLUX.2 Klein Cond Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
