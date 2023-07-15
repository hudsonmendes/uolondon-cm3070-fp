# Python Built-in Modules
from abc import abstractmethod
from typing import List, Type
from wave import Wave_read as Wave

# Third-Party Libraries
import torch
from torch.nn.utils.rnn import pad_sequence

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

    def forward(self, x: List[Wave]) -> torch.Tensor:
        """
        Create a representation based on the fixed size linear projection of the
        input tensor that represents the raw audio.

        :param x: the list of audio files that will be represented
        :return: matrix of tensors (batch_size, out_features)
        """
        y = self._preprocess(x)
        y = self.ff(y)
        y = y / torch.norm(y, dim=0)
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
            vecs.append(vec)
        return pad_sequence(vecs, batch_first=True)
