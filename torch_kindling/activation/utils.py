"""Utility functions for activation layers."""

from typing import Union

import torch.nn as nn


def get_activation(activation: Union[str, nn.Module]) -> nn.Module:
    """Get activation function from torch-kindling or torch.

    First attempts to get a custom activation from torch-kindling,
    then falls back to torch.nn activations.

    Args:
        activation: Either a string name ("relu", "gelu", "swish", etc.)
                   or an already instantiated nn.Module.

    Returns:
        Instantiated activation module.

    Raises:
        ValueError: If activation name not found in either torch-kindling or torch.
    """
    # If already a module, return it
    if isinstance(activation, nn.Module):
        return activation

    # Ensure string
    if not isinstance(activation, str):
        raise TypeError(f"activation must be str or nn.Module, got {type(activation)}")

    activation_lower = activation.lower().strip()

    # Mapping of common names to torch.nn classes
    activation_map = {
        "relu": "ReLU",
        "relu6": "ReLU6",
        "elu": "ELU",
        "selu": "SELU",
        "prelu": "PReLU",
        "rrelu": "RReLU",
        "gelu": "GELU",
        "silu": "SiLU",
        "swish": "SiLU",  # silu is also known as swish
        "mish": "Mish",
        "hardswish": "Hardswish",
        "hardsigmoid": "Hardsigmoid",
        "sigmoid": "Sigmoid",
        "tanh": "Tanh",
        "softsign": "Softsign",
        "softplus": "Softplus",
    }

    # Check if it's in the mapping
    if activation_lower in activation_map:
        try:
            activation_class = getattr(nn, activation_map[activation_lower])
            return activation_class()
        except AttributeError:
            pass

    # Try direct capitalization
    try:
        activation_class = getattr(nn, activation.capitalize())
        return activation_class()
    except AttributeError:
        pass

    # If not found, raise error with helpful message
    valid_options = ", ".join(sorted(activation_map.keys()))
    raise ValueError(f"Activation '{activation}' not found. " f"Available options: {valid_options}")
