# Python Built-in Modules
from abc import ABC
from typing import Type

# Third-Party Libraries
import torch

# Local Folders
from .erc_config import ERCConfig
from .erc_feedforward import ERCFeedForwardConfig, ERCFeedForwardModel


class ERCAudioEmbeddingType:
    WAVEFORM = "waveform"


class ERCAudioEmbeddings(ABC, torch.nn.Module):
    """
    Abstract class representing an Audio Feature Extraction model,
    responsible from extracting audio features from the input audio.
    """

    config: ERCConfig

    def __init__(self, config: ERCConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    @staticmethod
    def resolve_type_from(expression: str) -> Type["ERCAudioEmbeddings"]:
        if expression == ERCAudioEmbeddingType.WAVEFORM:
            return ERCRawAudioEmbeddings
        raise ValueError(f"Unknown audio embedding type: {expression}")


class ERCRawAudioEmbeddings(ERCAudioEmbeddings):
    """
    ERCRawAudioEmbeddings is a class that implements the
    Audio Feature Extraction model based on raw audio.
    """

    def __init__(self, config: ERCConfig) -> None:
        super().__init__(config)
        self.ff = ERCFeedForwardModel(
            in_features=config.audio_in_features,
            config=ERCFeedForwardConfig(
                hidden_size=config.audio_hidden_size,
                num_layers=config.audio_num_layers,
                dropout=config.audio_dropout,
                activation=config.audio_activation,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the audio embedding module.

        :param x: input tensor
        :return: output tensor
        """
        return self.ff(x)
