# Python Built-in Modules
from typing import Type

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig

# Local Folders
from .erc_config import ERCAudioEmbeddingType, ERCConfig
from .erc_emb import ERCEmbeddings
from .erc_feedforward import ERCFeedForwardConfig, ERCFeedForwardModel


class ERCAudioEmbeddings(ERCEmbeddings):
    """
    Abstract class representing an Audio Feature Extraction model,
    responsible from extracting audio features from the input audio.
    """

    def __init__(self, config: ERCConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

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

    hidden_size: int

    def __init__(self, config: ERCConfig) -> None:
        super().__init__(config)
        self.hidden_size = config.audio_hidden_size
        self.ff = ERCFeedForwardModel(
            in_features=config.audio_in_features,
            config=ERCFeedForwardConfig(
                hidden_size=config.audio_hidden_size,
                num_layers=config.audio_num_layers,
                dropout=config.audio_dropout,
                activation=config.audio_activation,
            ),
        )

    @property
    def out_features(self) -> int:
        return self.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the audio embedding module.

        :param x: input tensor
        :return: output tensor
        """
        return self.ff(x)
