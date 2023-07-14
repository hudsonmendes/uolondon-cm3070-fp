# Python Built-in Modules
from typing import Type

# Third-Party Libraries
import torch

# Local Folders
from .erc_config import ERCAudioEmbeddingType, ERCConfig
from .erc_emb import ERCEmbeddings
from .erc_feedforward import ERCFeedForwardConfig, ERCFeedForwardModel


class ERCAudioEmbeddings(ERCEmbeddings):
    """
    Abstract class representing an Audio Feature Extraction model,
    responsible from extracting audio features from the input audio.

    Example:
        >>> from hlm12erc.modelling.erc_emb_audio import ERCAudioEmbeddings
        >>> class ERCMyCustomAudioEmbeddings(ERCAudioEmbeddings):

    """

    def __init__(self, config: ERCConfig, *args, **kwargs) -> None:
        """
        Defines the constructor contract for the Audio Feature Extraction modules.
        """
        super().__init__(config, *args, **kwargs)

    @staticmethod
    def resolve_type_from(expression: str) -> Type["ERCAudioEmbeddings"]:
        """
        Resolves the Audio Feature Extraction module type from the given expression.

        :param expression: expression to resolve
        :return: resolved subtype of ERCAudioEmbeddings
        """
        if expression == ERCAudioEmbeddingType.WAVEFORM:
            return ERCRawAudioEmbeddings
        raise ValueError(f"Unknown audio embedding type: {expression}")


class ERCRawAudioEmbeddings(ERCAudioEmbeddings):
    """
    ERCRawAudioEmbeddings is a class that implements the
    Audio Feature Extraction model based on raw audio.

    Example:
        >>> from hlm12erc.modelling import ERCConfig, ERCAudioEmbeddingType
        >>> from hlm12erc.modelling.erc_emb_audio import ERCAudioEmbeddings
        >>> config = ERCConfig()
        >>> ERCAudioEmbeddings.resolve_type_from(ERCAudioEmbeddingType.WAVEFORM)(config)
    """

    hidden_size: int

    def __init__(self, config: ERCConfig) -> None:
        """
        Constructs the Audio Feature Extraction model based on raw audio.

        :param config: configuration for the model
        """
        super().__init__(config)
        self.hidden_size = config.audio_out_features
        self.ff = ERCFeedForwardModel(
            in_features=config.audio_in_features,
            config=ERCFeedForwardConfig(
                hidden_size=config.audio_out_features,
                num_layers=config.audio_num_layers,
                dropout=config.audio_dropout,
                activation=config.audio_activation,
            ),
        )

    @property
    def out_features(self) -> int:
        """
        Returns the number of output features of the audio embedding module.

        :return: number of output features
        """
        return self.hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a representation based on the fixed size linear projection of the
        input tensor that represents the raw audio.

        :param x: input tensor
        :return: output tensor
        """
        return self.ff(x)
