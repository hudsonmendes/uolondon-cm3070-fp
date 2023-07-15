# Python Built-in Modules
from dataclasses import dataclass
from typing import List, Optional

# Third-Party Libraries
import torch

# Local Folders
from .erc_config import ERCConfigFeedForwardLayer


class ERCFeedForward(torch.nn.Module):
    """
    Models a fully-connected feedforward network, that transforms
    the input through multiple layers of affine transformations,
    applying an activation function and dropout (optional) in between.
    """

    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, layers: Optional[List[ERCConfigFeedForwardLayer]]) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sequence = ERCFeedForward._layers_from(in_features=in_features, out_features=out_features, layers=layers)

    @staticmethod
    def _layers_from(
        in_features: int,
        out_features: int,
        layers: Optional[List[ERCConfigFeedForwardLayer]],
    ) -> torch.nn.Sequential:
        """
        Creates a sequence of affine transformations, applying an activation function
        and dropout (optional) in between, defined by a flexible configuration provided
        as a hyperparameter to the model.

        :param in_features: Number of input features
        :param out_features: Number of output features
        :param layers: List of ERCFeedForwardLayer objects, each representing a layer
        :return: A torch.nn.Sequential object
        """
        sequence: List[torch.nn.Module] = []
        if layers:
            for i, layer in enumerate(layers):
                effective_in_features = in_features if i == 0 else layers[i - 1].out_features
                effective_out_features = out_features if i == len(layers) - 1 else layer.out_features
                sequence.append(torch.nn.Linear(in_features=effective_in_features, out_features=effective_out_features))
                sequence.append(torch.nn.ReLU())
                if layer.dropout is not None:
                    sequence.append(torch.nn.Dropout(layer.dropout))
        else:
            sequence.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            sequence.append(torch.nn.ReLU())
        return torch.nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feedforward network.

        :param x: Input tensor with the loest level embedded representations to be transformed
        :return: Output tensor with the highest level embedded representations
        """
        return self.sequence(x)
