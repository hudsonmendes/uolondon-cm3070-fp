# Python Built-in Modules
from abc import abstractmethod
from typing import List, Type

# Third-Party Libraries
import torch
from torch.nn.functional import normalize as l2_norm

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig
from hlm12erc.modelling.erc_emb import ERCEmbeddings

# Local Folders
from .erc_config import ERCConfig, ERCFusionTechnique
from .erc_emb import ERCEmbeddings


class ERCFusion(ERCEmbeddings):
    """
    Contract for feature fusion networks.

    Example:
        >>> from hlm12erc.modelling.erc_fusion import ERCFusion, ERCFusionTechnique, ERCConfig
        >>> from hlm12erc.modelling.erc_emb import ERCEmbeddings
        >>> class ERCMyCustomFusion(ERCFusion):
        ...     pass
    """

    def __init__(self, embeddings: List[ERCEmbeddings], config: ERCConfig, *args, **kwargs) -> None:
        """
        Defines the constructor contract for fusion networks.

        :param embeddings: List of embeddings to be fused.
        :param config: Configuration object.
        """
        super(ERCFusion, self).__init__(config=config, *args, **kwargs)
        assert embeddings is not None
        assert config is not None

    @abstractmethod
    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """
        Defines the contract for the forward pass of the fusion network.

        :param x: List of tensors to be fused.
        :return: Fused tensor.
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

    def __init__(self, embeddings: List[ERCEmbeddings], config: ERCConfig) -> None:
        """
        Constructs a feature fusion network based on concatenation of vectors.

        :param embeddings: List of embeddings to be fused.
        :param config: Configuration object.
        """
        super().__init__(embeddings=embeddings, config=config)
        self.hidden_size = sum([e.out_features for e in embeddings])

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """
        Concatenates the input tensors along the feature dimension.

        :param x: List of tensors to be fused.
        :return: Fused tensor.
        """
        y = torch.cat(x, dim=1)
        y = l2_norm(y, p=2, dim=1)
        return y

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the embedding fusion.
        """
        return self.hidden_size


class ERCMultiheadedAttentionFusion(ERCFusion):
    """
    Feature Fusion Mechanism based on Multiheaded Attention.

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
    """

    hidden_size: int

    def __init__(self, embeddings: List[ERCEmbeddings], config: ERCConfig, *args, **kwargs) -> None:
        super().__init__(embeddings, config, *args, **kwargs)
        self.hidden_size = sum([e.out_features for e in embeddings])
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=config.fusion_attention_heads,
        )

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """
        Performs Multiheaded Attention on the input tensors, through
        concatenating the input, transforming the inputs using a
        multi-headed attention layer, and then performing the element-wise
        addition of the input and the attention output (residual connection).

        :param x: List of tensors to be fused, one or more for each modality, all with shape[0] == batch_size
        :return: Fused tensor, with dimensions (batch_size, concatenated_embedding_size)
        """
        y = torch.cat(x, dim=1)
        attn, _ = self.attn.forward(query=y, key=y, value=y)
        y = y + attn
        return y

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the embedding fusion,
        which is the same of the sum of the embedding dimensions, to allow for
        element-wise addition of the embeddings (residual connection)
        """
        return self.hidden_size
