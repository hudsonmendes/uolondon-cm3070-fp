# Python Built-in Modules
from abc import abstractmethod
from typing import List

# Third-Party Libraries
import torch
import torchvision
from PIL.Image import Image
from torch.nn.functional import normalize as l2_norm

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
        Initializes the ERCResNet50VisualEmbeddings class, loading the preprocessing
        transformation routine and the ResNet50 model.

        The preprocessing transformation routine normalises the input images using
        the meand and standard deviation calculated on the images of the training
        set, calculated as part of the mlops notebook, in the "Measures of Spread"
        section, by the function `calculate_measures_of_spread` .

        The resnet50 model is loaded from the torchvision library, and the final
        fully connected layer is replaced by an identity layer, so that the output
        of the model is the output of the final convolutional layer.

        :param config: The configuration for the ERC model.
        """
        super().__init__(config)
        self.preprocessor = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.Lambda(lambda x: torchvision.transforms.functional.crop(x, 0, 0, 256, 721)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.2706, 0.2010, 0.1914], std=[0.1857, 0.1608, 0.1667]),
            ]
        )
        self.resnet50 = torch.hub.load("pytorch/vision:v0.6.0", "resnet50", pretrained=True)
        self.resnet50.eval()
        self.resnet50.fc = torch.nn.Identity()

    @property
    def out_features(self) -> int:
        """
        Returns the number of output features of the visual embedding layer.

        :return: The number of output features of the visual embedding layer.
        """
        return 2048

    def forward(self, x: List[Image]) -> torch.Tensor:
        """
        Performs a forward pass through the ResNet50 visual embedding layer.

        :param x: The input tensor.
        :return: The output tensor with the restnet50 embedded representation.
        """
        y = torch.stack([self.preprocessor(xi) for xi in x], dim=0)
        y = self.resnet50(y)
        y = l2_norm(y, p=2, dim=1)
        return y
