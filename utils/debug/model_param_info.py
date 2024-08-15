import torch
from torch import nn


def print_model_params_info(model: nn.Module):
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(name, f": {param.abs().mean():.4f}, grad: {param.grad.abs().mean():.4f}")
        else:
            print(name, f": {param.abs().mean():.4f}")
    print("-" * 100)
    for name, buffer in model.named_buffers():
        print(name, f": {buffer.abs().mean():.4f}")
    print("-" * 100)


def register_print_activation_hook(model: nn.Module):
    def hook(cur_module: nn.Module, inputs):
        print(cur_module.__class__.__name__, f": {len(inputs)}, {inputs[0].abs().mean():.4f}")

    for module in model.modules():
        module.register_forward_pre_hook(hook)
