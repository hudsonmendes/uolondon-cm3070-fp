# Python Built-in Modules
from dataclasses import dataclass
from typing import List

# Third-Party Libraries
import torch


@dataclass(frozen=True)
class ERCFeedForwardConfig:
    """
    Provides the configuration for the feedforward network.
    """

    hidden_size: int
    num_layers: int
    dropout: float
    activation: str


class ERCFeedForwardActivation:
    RELU = "relu"


class ERCFeedForwardModel(torch.nn.Module):
    """
    Models a fully-connected feedforward network, that transforms
    the input through multiple layers of affine transformations,
    applying an activation function and dropout (optional) in between.
    """

    def __init__(self, in_features: int, config: ERCFeedForwardConfig) -> None:
        """
        Creates a feedforward network based on the given ERCConfig.

        :param in_features: number of input features that will enter the FFN
        :param config: configuration for the FFN including layers, dropout, activation
        """
        super().__init__()
        self.ff = ERCFeedForwardModel._create_feedforward_based_on(in_features, config)

    @staticmethod
    def _create_feedforward_based_on(in_features: int, config: ERCFeedForwardConfig) -> torch.nn.Module:
        """
        Creates a feedforward network based on the given ERCConfig.

        :param in_features: number of input features that will enter the FFN
        :param config: configuration for the FFN including layers, dropout, activation
        :return: a fully-connected feedforward network
        """
        layers: List[torch.nn.Module] = []
        for i in range(config.num_layers):
            # linear layer
            layer_in_features = in_features if i == 0 else config.hidden_size
            layers.append(torch.nn.Linear(layer_in_features, config.hidden_size))
            # activation
            activation_type = ERCFeedForwardModel.resolve_activation_from(config.activation)
            layers.append(activation_type())
            # dropout
            if config.dropout > 0:
                layers.append(torch.nn.Dropout(config.dropout))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def resolve_activation_from(expression: str) -> torch.nn.Module:
        if expression == ERCFeedForwardActivation.RELU:
            return torch.nn.ReLU()
        raise ValueError(f"The activation {expression} is not supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the feedforward network.

        :param x: input tensor
        :return: output tensor
        """
        return self.ff(x)
