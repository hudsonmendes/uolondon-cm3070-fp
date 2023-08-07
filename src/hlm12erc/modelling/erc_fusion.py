# Python Built-in Modules
from abc import abstractmethod
from typing import List, Type

# Third-Party Libraries
import torch
from torch.nn.functional import normalize as l2_norm

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
        raise ValueError(f"The fusion '{expression}' is not supported.")


class ERCConcatFusion(ERCFusion):
    """
    Simple implementation of feature fusion based on concatenation of vectors.
    """

    concatenated_embedding_dims: int

    def __init__(self, embeddings: List[ERCEmbeddings], config: ERCConfig) -> None:
        """
        Constructs a feature fusion network based on concatenation of vectors.

        :param embeddings: List of embeddings to be fused.
        :param config: Configuration object.
        """
        super().__init__(embeddings=embeddings, config=config)
        self.concatenated_embedding_dims = sum([e.out_features for e in embeddings])

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """
        Concatenates the input tensors along the feature dimension.

        :param x: List of tensors to be fused.
        :return: Fused tensor.
        """
        y = torch.concat(x, dim=1)
        y = l2_norm(y, p=2, dim=1)
        return y

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the embedding fusion.
        """
        return self.concatenated_embedding_dims
