# Python Built-in Modules
from abc import abstractmethod
from typing import List, Type
from wave import Wave_read as Wave

# Third-Party Libraries
import torch
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
    def forward(self, x: List[Wave]) -> torch.Tensor:
        """
        When implemented, this method should receive a list of audio files and
        return a matrix of tensors (batch_size, out_features).

        :param x: list of audio files
        :return: matrix of tensors (batch_size, out_features)
        """
        raise NotImplementedError("The method 'forward' must be implemented.")

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

    in_features: int

    def __init__(self, config: ERCConfig) -> None:
        """
        Constructs the Audio Feature Extraction model based on raw audio.

        :param config: configuration for the model
        """
        super().__init__(config)
        self.in_features = config.audio_in_features
        self.ff = ERCFeedForward(
            in_features=config.audio_in_features,
            layers=[
                ERCConfigFeedForwardLayer(out_features=config.audio_out_features * 3, dropout=0.1),
                ERCConfigFeedForwardLayer(out_features=config.audio_out_features * 2, dropout=0.1),
                ERCConfigFeedForwardLayer(out_features=config.audio_out_features, dropout=0.1),
            ],
        )

    @property
    def out_features(self) -> int:
        """
        Returns the number of features that the audio embedding will return,
        after the transformations that projec the original raw audio into
        a fixed size vector.

        :return: number of features
        """
        return self.ff.out_features

    def forward(self, x: List[Wave]) -> torch.Tensor:
        """
        Create a representation based on the fixed size linear projection of the
        input tensor that represents the raw audio.

        :param x: the list of audio files that will be represented
        :return: matrix of tensors (batch_size, out_features)
        """
        y = self._preprocess(x)
        y = self.ff(y)
        y = l2_norm(y, p=2, dim=1)
        return y

    def _preprocess(self, x: List[Wave]) -> torch.Tensor:
        """
        Pre-processes the batch of audio files into a single
        matrix containing the stacked batch of audio tensors.

        :param x: the list of audio files that will be represented
        :return: matrix of tensors (batch_size, len(samples))
        """
        vecs: List[torch.Tensor] = []
        dtype_map = {1: torch.int8, 2: torch.int16, 4: torch.int32}
        for wave_file in x:
            data = wave_file.readframes(wave_file.getnframes())
            dtype = dtype_map[wave_file.getsampwidth()]
            samples = torch.frombuffer(data, dtype=dtype).float()
            samples = samples.reshape(-1, wave_file.getnchannels())
            vec = samples.flatten()
            vec_len = vec.shape[0]
            if vec_len < self.in_features:
                padding_len = self.in_features - vec_len
                padding = torch.zeros(padding_len)
                vec = torch.cat([vec, padding])
            elif vec_len > self.in_features:
                vec = vec[: self.in_features]
            vecs.append(vec)
        return torch.stack(vecs)
