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
        assert config is not None
        assert config.audio_in_features is not None
        assert config.audio_out_features is not None
        self.config = config
        self.in_features = config.audio_in_features
        self.mha, self.layer_norm = None, None
        if config.audio_attention_heads_degree is not None:
            dims_out = config.audio_out_features
            heads_candidates = [i for i in range(1, dims_out + 1) if dims_out % i == 0]
            heads_i = config.audio_attention_heads_degree - 1
            heads_count = heads_candidates[heads_i] if heads_i < len(heads_candidates) else heads_candidates[-1]
            self.mha = torch.nn.MultiheadAttention(embed_dim=dims_out, num_heads=heads_count)
            self.layer_norm = torch.nn.LayerNorm(dims_out)
        self.ff = ERCFeedForward(
            in_features=config.audio_in_features,
            layers=[
                ERCConfigFeedForwardLayer(
                    out_features=config.audio_out_features * 3,
                    dropout=0.1,
                ),
                ERCConfigFeedForwardLayer(
                    out_features=config.audio_out_features * 2,
                    dropout=0.1,
                ),
                ERCConfigFeedForwardLayer(
                    out_features=config.audio_out_features,
                    dropout=0.1,
                ),
            ],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a representation based on the fixed size linear projection of the
        input tensor that represents the raw audio.

        :param x: stacked vectors representing audio waveforms
        :return: matrix of tensors (batch_size, out_features)
        """
        # transform the raw audio into a lower dimensional representation
        y = self.ff(x)
        # if multi-headed attention is enabled, apply it to the output
        # and add the attention output to the original output (residual connection)
        # and normalize the output vector to have unit norm
        if self.mha is not None:
            attn, _ = self.mha(y, y, y)[0]
            y = y + attn
            y = self.layer_norm(y)
        # normalize the output vector to have unit norm
        return l2_norm(y, p=2, dim=1)

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the embedding transformation.
        """
        return self.config.audio_out_features


class ERCWave2Vec2Embeddings(ERCAudioEmbeddings):
    """
    ERCWave2Vec2Embeddings is a class that implements the
    Audio Feature Extraction model using the pretrained Wave2Vec2 model,
    by concatenating the mean and max pooling of the hidden states.

    References:
        >>> Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli.
        ... 2020. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
        ... Representations. In Advances in Neural Information Processing Systems,
        ... Curran Associates, Inc., 12449–12460. Retrieved
        ... from https://proceedings.neurips.cc/paper_files/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf

        >>> Minghua Zhang, Yunfang Wu, Weikang Li, and Wei Li. 2018. Learning
        ... Universal Sentence Representations with Mean-Max Attention Autoencoder.
        ... In Proceedings of the 2018 Conference on Empirical Methods in Natural
        ... Language Processing, 4514–4523.

    Example:
        >>> from hlm12erc.modelling import ERCConfig, ERCAudioEmbeddingType
        >>> from hlm12erc.modelling.erc_emb_audio import ERCAudioEmbeddings
        >>> config = ERCConfig()
        >>> ERCAudioEmbeddings.resolve_type_from(ERCAudioEmbeddingType.WAV2VEC2)(config)
    """

    in_features: int
    hidden_size: int
    wav2vec2: transformers.Wav2Vec2Model
    fc: torch.nn.Linear | None

    def __init__(self, config: ERCConfig) -> None:
        """
        Constructs the Audio Feature Extraction model based on raw audio.

        :param config: configuration for the model
        """
        super(ERCWave2Vec2Embeddings, self).__init__(config=config)
        assert config is not None
        self.config = config
        self.in_features = config.audio_in_features
        self.hidden_size = config.audio_out_features
        self.wav2vec2 = transformers.Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        # determine the hidden size of this layer which may either be set by config
        # or twice the size of the wav2vec2 hidden size (mean+max pooling)
        wav2vec2_meanmax_dims = self.wav2vec2.config.hidden_size * 2
        if self.hidden_size is None or self.hidden_size <= 0:
            self.hidden_size = wav2vec2_meanmax_dims  # mean+max pooling
        # only create project _if_ the output size is different from the hidden size
        self.fc = None
        if wav2vec2_meanmax_dims != self.hidden_size:
            self.fc = torch.nn.Linear(wav2vec2_meanmax_dims, self.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a representation based on the last hidden state of the pretrained
        Wave2Vec2 model concatenating the mean and max pooling of the hidden states.

        :param x: stacked vectors representing audio waveforms
        :return: matrix of tensors (batch_size, out_features)
        """

        # concatenate the mean and max pooling of the hidden states
        # to generate a fixed size vector that can be used as input
        # to the classifier
        h = self.wav2vec2(x).last_hidden_state
        h_mean = torch.mean(h, dim=1)
        h_max = torch.max(h, dim=1)[0]
        y = torch.cat((h_mean, h_max), dim=1)

        # only projects if the output size is different from the hidden size
        if self.fc is not None and self.hidden_size != y.size(dim=1):
            y = self.fc(y)

        # normalize the output vector to have unit norm
        return l2_norm(y, p=2, dim=1)

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the wav2vec2 model.
        """
        return self.hidden_size
