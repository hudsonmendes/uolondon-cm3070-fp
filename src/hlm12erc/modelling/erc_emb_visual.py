# Python Built-in Modules
from abc import abstractmethod
from typing import List

# Third-Party Libraries
import torch
from PIL.Image import Image
from transformers import AutoImageProcessor, ResNetModel

# Local Folders
from .erc_config import ERCConfig, ERCVisualEmbeddingType
from .erc_emb import ERCEmbeddings


class ERCVisualEmbeddings(ERCEmbeddings):
    """
    ERCVisualEmbeddings is an abstract class that defines the interface for visual embedding layers.

    Examples:
        >>> from hlm12erc.modelling.erc_emb_text import ERCTextEmbeddings
        >>> class ERCMyCustomTextEmbeddings(ERCTextEmbeddings):
    """

    @abstractmethod
    def forward(self, x: List[str]) -> torch.Tensor:
        """
        When implemented, this method should receive a list of images and
        return a matrix of tensors (batch_size, out_features).
        """
        raise NotImplementedError("The method 'forward' must be implemented.")

    @staticmethod
    def resolve_type_from(expression: str) -> type["ERCVisualEmbeddings"]:
        if expression == ERCVisualEmbeddingType.RESNET50:
            return ERCResNet50VisualEmbeddings
        raise ValueError(f"The visual embedding '{expression}' is not supported.")


class ERCResNet50VisualEmbeddings(ERCVisualEmbeddings):
    """
    ERCResNet50VisualEmbeddings is a class that implements the visual embedding
    layer using ResNet50 and simply returning the output of the final layer. The
    embeddings transformation is based on the paper:
    "Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, 770–778."

    Example:
        >>> from hlm12erc.modelling import ERCConfig, ERCVisualEmbeddingType
        >>> from hlm12erc.modelling.erc_emb_visual import ERCVisualEmbeddings
        >>> config = ERCConfig()
        >>> ERCVisualEmbeddings.resolve_type_from(ERCVisualEmbeddingType.RESNET50)(config)
    """

    def __init__(self, config: ERCConfig):
        """
        Initializes the ERCResNet50VisualEmbeddings class.

        :param config: The configuration for the ERC model.
        """
        super().__init__(config)
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.resnet50 = ResNetModel.from_pretrained("microsoft/resnet-50")

    @property
    def out_features(self) -> int:
        """
        Returns the number of output features of the visual embedding layer.

        :return: The number of output features of the visual embedding layer.
        """
        return 1024

    def forward(self, x: List[Image]) -> torch.Tensor:
        """
        Performs a forward pass through the ResNet50 visual embedding layer.

        :param x: The input tensor.
        :return: The output tensor with the restnet50 embedded representation.
        """
        y = self.processor(x, return_tensors="pt")
        y = self.resnet50(**y).last_hidden_state
        return y
