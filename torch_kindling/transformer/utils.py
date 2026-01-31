"""Utility functions for transformer layers."""

from typing import Union

import torch.nn as nn

from torch_kindling.activation import get_activation


def build_feedforward(
    d_model: int, d_ff: int, activation: Union[str, nn.Module], dropout: float
) -> nn.Sequential:
    """Build a feedforward network for transformer layers.

    Args:
        d_model (int): Model dimension.
        d_ff (int): Hidden dimension of feedforward.
        activation (str or nn.Module): Activation function name or instantiated module.
        dropout (float): Dropout rate.

    Returns:
        nn.Sequential: Feedforward module.
    """
    act = get_activation(activation)

    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        act,
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
    )
