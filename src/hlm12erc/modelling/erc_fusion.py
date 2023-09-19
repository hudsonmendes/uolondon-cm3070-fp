# Python Built-in Modules
import logging
from abc import abstractmethod
from typing import List, Tuple, Type

# Third-Party Libraries
import torch
from torch.nn.functional import normalize as l2_norm

# Local Folders
from .erc_config import ERCConfig, ERCFusionTechnique
from .erc_emb import ERCEmbeddings
from .erc_emb_audio import ERCAudioEmbeddings
from .erc_emb_text import ERCTextEmbeddings
from .erc_emb_visual import ERCVisualEmbeddings

logger = logging.getLogger(__name__)


class ERCFusion(ERCEmbeddings):
    """
    Contract for feature fusion networks.

    Example:
        >>> from hlm12erc.modelling.erc_fusion import ERCFusion, ERCFusionTechnique, ERCConfig
        >>> from hlm12erc.modelling.erc_emb import ERCEmbeddings
        >>> class ERCMyCustomFusion(ERCFusion):
        ...     pass
    """

    def __init__(
        self,
        embeddings: Tuple[ERCTextEmbeddings | None, ERCVisualEmbeddings | None, ERCAudioEmbeddings | None],
        config: ERCConfig,
        *args,
        **kwargs,
    ) -> None:
        """
        Defines the constructor contract for fusion networks.

        :param embeddings: Tuple with the embeddings for the 3 modalities
        :param config: Configuration object.
        """
        super(ERCFusion, self).__init__(config=config, *args, **kwargs)
        assert embeddings is not None
        assert config is not None

    @abstractmethod
    def forward(
        self,
        x_text: torch.Tensor | None,
        x_visual: torch.Tensor | None,
        x_audio: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        Defines the contract for the forward pass of the fusion network.

        :param x: List of tensors to be fused.
        :return: Tuple with fused tensor and attention weights.
        """
        raise NotImplementedError("Abstract method not implemented")

    @staticmethod
    def resolve_type_from(expression: str) -> Type["ERCFusion"]:
        if expression == ERCFusionTechnique.CONCATENATION:
            return ERCConcatFusion
        elif expression == ERCFusionTechnique.MULTI_HEADED_ATTENTION:
            return ERCMultiheadedAttentionFusion
        raise ValueError(f"The fusion '{expression}' is not supported.")


class ERCConcatFusion(ERCFusion):
    """
    Simple implementation of feature fusion based on concatenation of vectors.
    """

    hidden_size: int

    def __init__(
        self,
        embeddings: Tuple[ERCTextEmbeddings | None, ERCVisualEmbeddings | None, ERCAudioEmbeddings | None],
        config: ERCConfig,
    ) -> None:
        """
        Constructs a feature fusion network based on concatenation of vectors.

        :param embeddings: Tuple with the embeddings for the 3 modalities
        :param config: Configuration object.
        """
        super().__init__(embeddings=embeddings, config=config)
        self.hidden_size = sum([e.out_features for e in embeddings if e is not None])

    def forward(
        self,
        x_text: torch.Tensor | None,
        x_visual: torch.Tensor | None,
        x_audio: torch.Tensor | None,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        Concatenates the input tensors along the feature dimension.

        :param x: List of tensors to be fused.
        :return: Tuple with fused tensor and attention weights.
        """
        x = [x_modal for x_modal in (x_text, x_visual, x_audio) if x_modal is not None]
        y = torch.cat(x, dim=1)
        y = l2_norm(y, p=2, dim=1)
        return y, None

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the embedding fusion.
        """
        return self.hidden_size


class ERCMultiheadedAttentionFusion(ERCFusion):
    """
    Feature Fusion Mechanism based on Multiheaded Attention.

    Due to the O(n^2) complexity of the attention mechanism, the input dimension
    of each modality has to be reduced to a smaller dimensionality, which is
    done through projection over linear layer with a smaller output dimension,
    but with dimensionality proportional to the original input dimension.
    These mapped embeddings are then concatenated and fed into the attention
    mechanism, which outputs a tensor of the same dimensionality of the input.

    Using the Multiheaded Attention as a form of feature fusion is inspired
    by the following paper:
    >>> Vishal Chudasama, Purbayan Kar, Ashish Gudmalwar, Nirmesh Shah, Pankaj
    ... Wasnik, and Naoyuki Onoe. 2022. M2FNet: Multi-modal Fusion Network for
    ... Emotion Recognition in Conversation. In 2022 IEEE/CVF Conference on Computer
    ... Vision and Pattern Recognition Workshops (CVPRW), 4651–4660. DOI:https://doi.org/10.1109/CVPRW56347.2022.00511

    And using the Multiheaded Attention with a residual connection takes inspiration
    in the following paper:
    >>> Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    ... Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is
    ... All you Need. In Advances in Neural Information Processing Systems,
    ... Curran Associates, Inc. Retrieved from https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf

    Finally, past the residual connection, layer normalisation is applied, as
    suggested in the following paper:
    >>> Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. 2016. Layer Normalization.
    ... Retrieved from https://arxiv.org/abs/1607.06450
    """

    def __init__(
        self,
        embeddings: Tuple[ERCTextEmbeddings | None, ERCVisualEmbeddings | None, ERCAudioEmbeddings | None],
        config: ERCConfig,
    ) -> None:
        """
        Constructs a feature fusion network based on Multiheaded Attention.

        :param embeddings: Tuple with the embeddings for the 3 modalities
        :param config: Configuration object.
        """
        super().__init__(embeddings, config)

        # prepare the configuration
        embed_text, embed_visual, embed_audio = embeddings
        assert isinstance(config.fusion_attention_heads_degree, int)
        assert isinstance(config.fusion_out_features, int)
        assert embeddings[0] is None or isinstance(embeddings[0], ERCTextEmbeddings)
        assert embeddings[1] is None or isinstance(embeddings[1], ERCVisualEmbeddings)
        assert embeddings[2] is None or isinstance(embeddings[2], ERCAudioEmbeddings)

        # prepare the attention network
        dims_src_all = [e.out_features if e else None for e in embeddings]
        dims_src_concat = sum([(dim or 0) for dim in dims_src_all])
        dims_src_ratios = [(dim / dims_src_concat if dim else None) for dim in dims_src_all]
        dims_dst_all = [(int(config.fusion_out_features * ratio) if ratio else None) for ratio in dims_src_ratios]
        dims_dst_diff = config.fusion_out_features - sum([dim or 0 for dim in dims_dst_all if dim])
        dims_dst_max = max([dim for dim in dims_dst_all if dim])
        dims_dst_max_i = dims_dst_all.index(dims_dst_max)
        dims_dst_all[dims_dst_max_i] = (dims_dst_all[dims_dst_max_i] or 0) + dims_dst_diff
        dims_dst_text, dims_dst_visual, dims_dst_audio = dims_dst_all
        dims_out = sum([dim or 0 for dim in dims_dst_all])
        heads_candidates = [i for i in range(1, dims_out + 1) if dims_out % i == 0]
        heads_i = config.fusion_attention_heads_degree - 1
        heads_count = heads_candidates[heads_i] if heads_i < len(heads_candidates) else heads_candidates[-1]
        self.fc_text, self.fc_visual, self.fc_audio = (None, None, None)
        if embed_text is not None and dims_dst_text is not None:
            self.fc_text = torch.nn.Linear(in_features=embed_text.out_features, out_features=dims_dst_text)
        if embed_visual is not None and dims_dst_visual is not None:
            self.fc_visual = torch.nn.Linear(in_features=embed_visual.out_features, out_features=dims_dst_visual)
        if embed_audio is not None and dims_dst_audio is not None:
            self.fc_audio = torch.nn.Linear(in_features=embed_audio.out_features, out_features=dims_dst_audio)
        logger.warn(f"FUSION: Attention Heads={heads_count}, for Embed Dim={dims_out} (from {dims_src_concat})")
        self.attn = torch.nn.MultiheadAttention(embed_dim=dims_out, num_heads=heads_count)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape=dims_out)

    def forward(
        self,
        x_text: torch.Tensor | None,
        x_visual: torch.Tensor | None,
        x_audio: torch.Tensor | None,
    ) -> (torch.Tensor, torch.Tensor | None):
        """
        Performs Multiheaded Attention on the input tensors,
        through concatenating the input, transforming the inputs
        using a multi-headed attention layer, and then performing
        the element-wise addition of the input and the attention
        output (residual connection).

        Each input X (for each modality) is projected over a linear
        representation of output dimensionality proportional to the
        input dimensionality.

        These projected representations are then concatenated and fed
        into the attention mechanism, which outputs a tensor of the
        same dimensionality of the input.

        Finally, the output of the attention mechanism is added to
        the input (residual connection), and layer normalisation is
        applied.

        :param x_text: Tensor with the text embeddings, with dimensions (batch_size, text_embedding_size)
        :param x_visual: Tensor with the visual embeddings, with dimensions (batch_size, visual_embedding_size)
        :param x_audio: Tensor with the audio embeddings, with dimensions (batch_size, audio_embedding_size)
        :return: Tuple with fused tensor and attention weights.
        """
        # dimensionality reduction through mapping using a linear layer
        x_all = []
        if self.fc_text:
            x_all.append(self.fc_text(x_text))
        if self.fc_visual:
            x_all.append(self.fc_visual(x_visual))
        if self.fc_audio:
            x_all.append(self.fc_audio(x_audio))
        # concatenate the mapped embeddings
        x = torch.cat(x_all, dim=1)
        # perform the attention mechanism
        attn, _ = self.attn.forward(query=x, key=x, value=x)
        # residual connection
        y = x + attn
        # layer normalisation
        y = self.layer_norm(y)
        # output
        return y, attn

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the embedding fusion,
        which is the same of the sum of the embedding dimensions, to allow for
        element-wise addition of the embeddings (residual connection)
        """
        return self.attn.embed_dim
