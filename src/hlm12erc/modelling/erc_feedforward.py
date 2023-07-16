# Python Built-in Modules
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
        self.sequence = torch.nn.Sequential(
            *ERCFeedForwardLayersFactory(layers=layers).create(
                in_features=in_features,
                out_features=out_features,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feedforward network.

        :param x: Input tensor with the loest level embedded representations to be transformed
        :return: Output tensor with the highest level embedded representations
        """
        return self.sequence(x)


class ERCFeedForwardLayersFactory:
    """
    Factory class to create a sequence of affine transformations, applying an activation function
    and dropout (optional) in between, defined by a flexible configuration provided as a
    hyperparameter to the model.

    Example:
        >>> factory = ERCFeedForwardLayersFactory(layers=[
        ...     ERCConfigFeedForwardLayer(out_features=32, dropout=0.1),
        ...     ERCConfigFeedForwardLayer(out_features=16, dropout=0.1),
        ...     ERCConfigFeedForwardLayer(out_features=8, dropout=0.1),
        ...     ERCConfigFeedForwardLayer(out_features=4, dropout=0.1),
        ... ])
        >>> factory.create(in_features=64, out_features=4)
    """

    def __init__(self, layers: Optional[List[ERCConfigFeedForwardLayer]]) -> None:
        self.layers = layers

    def create(self, in_features: int, out_features: int) -> List[torch.nn.Module]:
        """
        Creates a list of layers corresponding to the configuration passed in the constructor,
        and to the `in_features` and `out_features` passed as arguments. In the absence of a
        configuration, a single layer is created, with the `in_features` and `out_features`.

        :param in_features: Number of input features
        :param out_features: Number of output features
        :param layers: List of ERCFeedForwardLayer objects, each representing a layer
        :return: A torch.nn.Sequential object
        """
        sequence: List[torch.nn.Module] = []
        if self.layers:
            # ensure that the last layer has no out_features
            assert self.layers[-1].out_features is None, "The `out_features` of the last layer must be `None`"

            # the number of input features of the first layer is the number of input features of the model
            last_out_features = effective_out_features = in_features
            for i, layer in enumerate(self.layers):
                # the `out_feature` of intermediate layers (hidden dims) is either:
                # - the `out_features` of the feedforward, if it's the last layer
                # - the `out_features` of the layer, if it's not the last layer and the layer has an `out_features`
                # - the `out_features` of the previous layer, if it's not the last layer and the layer has no
                if i == len(self.layers) - 1:
                    effective_out_features = out_features
                elif layer.out_features:
                    effective_out_features = layer.out_features
                sequence.append(torch.nn.Linear(in_features=last_out_features, out_features=effective_out_features))
                # in this implementation, we default the activation to ReLU
                sequence.append(torch.nn.ReLU())
                # dropout is define optionally
                if layer.dropout is not None:
                    sequence.append(torch.nn.Dropout(layer.dropout))
                # if the layer has an `out_features`, we update the `last_out_features` to the `out_features`
                if layer.out_features:
                    last_out_features = layer.out_features
        else:
            # if no layers are provided, we default to a single layer with the same `in_features` and `out_features`
            # so we still generate a representation, but through a shallow network.
            sequence.append(torch.nn.Linear(in_features=in_features, out_features=out_features))
            sequence.append(torch.nn.ReLU())
        return sequence
