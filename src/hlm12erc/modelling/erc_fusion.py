# Python Built-in Modules
from abc import abstractmethod
from typing import List, Type

# Third-Party Libraries
import torch

# Local Folders
from .erc_config import ERCConfig, ERCFusionTechnique
from .erc_emb import ERCEmbeddings


class ERCFusion(ERCEmbeddings):
    def __init__(self, embeddings: List[ERCEmbeddings], config: ERCConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        assert embeddings is not None
        assert config is not None

    @abstractmethod
    def forward(self, *x: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Abstract method not implemented")

    @staticmethod
    def resolve_type_from(expression: str) -> Type["ERCFusion"]:
        if expression == ERCFusionTechnique.STACKED:
            return ERCStackedFusion
        raise ValueError(f"The fusion '{expression}' is not supported.")


class ERCStackedFusion(ERCFusion):
    stacked_embedding_dims: int

    def __init__(self, embeddings: List[ERCEmbeddings], config: ERCConfig) -> None:
        super().__init__(embeddings=embeddings, config=config)
        self.stacked_embedding_dims = sum([e.out_features for e in embeddings])

    @property
    def out_features(self) -> int:
        return self.stacked_embedding_dims

    def forward(self, *x: List[torch.Tensor]) -> torch.Tensor:
        return torch.stack(tuple(*x))
