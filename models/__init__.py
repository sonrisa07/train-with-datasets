import torch.nn

from .vgg import *


def generate_model(model_name) -> torch.nn.Module:
    if model_name[0:3] == "VGG":
        return VGG(3, 10, model_name)
    else:
        if model_name in __all__[1:]:
            return globals()[model_name]()


__all__ = [
    "generate_model",
    "VGG",
]
