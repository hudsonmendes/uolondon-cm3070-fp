# Python Built-in Modules
from abc import abstractmethod
from typing import Callable, Optional, Type

# Third-Party Libraries
import torch
import transformers
from torch.nn.functional import normalize as l2_norm

# Local Folders
from .erc_config import ERCAudioEmbeddingType, ERCConfig, ERCConfigFeedForwardLayer
from .erc_emb import ERCEmbeddings
from .erc_feedforward import ERCFeedForward


class ERCAudioEmbeddings(ERCEmbeddings):
    """
    Abstract class representing an Audio Feature Extraction model,
    responsible from extracting audio features from the input audio.

    Example:
        >>> from hlm12erc.modelling.erc_emb_audio import ERCAudioEmbeddings
        >>> class ERCMyCustomAudioEmbeddings(ERCAudioEmbeddings):

    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        When implemented, this method should receive a list of audio files and
        return a matrix of tensors (batch_size, out_features).

        :param x: stacked vectors representing audio waveforms
        :return: matrix of tensors (batch_size, out_features)
        """
        raise NotImplementedError("The method 'forward' must be implemented.")

    @staticmethod
    def resolve_type_from(
        expression: str,
    ) -> Type["ERCAudioEmbeddings"] | Callable[[ERCConfig], Optional["ERCAudioEmbeddings"]]:
        """
        Resolves the Audio Feature Extraction module type from the given expression.

        :param expression: expression to resolve
        :return: resolved subtype of ERCAudioEmbeddings
        """
        if expression == ERCAudioEmbeddingType.WAVEFORM:
            return ERCRawAudioEmbeddings
        elif expression == ERCAudioEmbeddingType.WAV2VEC2:
            return ERCWave2Vec2Embeddings
        elif expression == ERCAudioEmbeddingType.NONE:
            return lambda _: None
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

    in_features: int

    def __init__(self, config: ERCConfig) -> None:
        """
        Constructs the Audio Feature Extraction model based on raw audio.

        :param config: configuration for the model
        """
        super(ERCRawAudioEmbeddings, self).__init__(config=config)
        self.config = config
        self.in_features = config.audio_in_features
        self.ff = ERCFeedForward(
            in_features=config.audio_in_features,
            layers=[
                ERCConfigFeedForwardLayer(out_features=config.audio_out_features * 3, dropout=0.1),
                ERCConfigFeedForwardLayer(out_features=config.audio_out_features * 2, dropout=0.1),
                ERCConfigFeedForwardLayer(out_features=config.audio_out_features, dropout=0.1),
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a representation based on the fixed size linear projection of the
        input tensor that represents the raw audio.

        :param x: stacked vectors representing audio waveforms
        :return: matrix of tensors (batch_size, out_features)
        """
        y = self.ff(x)
        y = l2_norm(y, p=2, dim=1)
        return y

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the embedding transformation.
        """
        return self.config.audio_out_features


class ERCWave2Vec2Embeddings(ERCAudioEmbeddings):
    """
    ERCWave2Vec2Embeddings is a class that implements the
    Audio Feature Extraction model using the pretrained Wave2Vec2 model.

    Example:
        >>> from hlm12erc.modelling import ERCConfig, ERCAudioEmbeddingType
        >>> from hlm12erc.modelling.erc_emb_audio import ERCAudioEmbeddings
        >>> config = ERCConfig()
        >>> ERCAudioEmbeddings.resolve_type_from(ERCAudioEmbeddingType.WAV2VEC2)(config)
    """

    in_features: int

    def __init__(self, config: ERCConfig) -> None:
        """
        Constructs the Audio Feature Extraction model based on raw audio.

        :param config: configuration for the model
        """
        super(ERCWave2Vec2Embeddings, self).__init__(config=config)
        self.config = config
        self.in_features = config.audio_in_features
        self.wav2vec2 = transformers.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a representation based using the last hidden state of the
        pretrained Wave2Vec2 model.

        :param x: stacked vectors representing audio waveforms
        :return: matrix of tensors (batch_size, out_features)
        """
        y = self.wav2vec2(x).last_hidden_state
        y = l2_norm(y, p=2, dim=1)
        return y

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the wav2vec2 model.
        """
        return self.wav2vec2.config.hidden_size
