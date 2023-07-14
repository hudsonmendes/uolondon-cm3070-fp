# Third-Party Libraries
import torch

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

    @staticmethod
    def resolve_type_from(expression: str) -> type["ERCVisualEmbeddings"]:
        if expression == ERCVisualEmbeddingType.RESNET50:
            return ERCResNet50VisualEmbeddings
        raise ValueError(f"The visual embedding '{expression}' is not supported.")


class ERCResNet50VisualEmbeddings(ERCVisualEmbeddings):
    """
    ERCResNet50VisualEmbeddings is a class that implements the visual embedding
    layer using ResNet50 and simply returning the output of the final layer.

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
        self.resnet50 = torch.hub.load("pytorch/vision:v0.6.0", "resnet50", pretrained=True)
        self.resnet50.eval()
        self.resnet50.fc = torch.nn.Identity()

    @property
    def out_features(self) -> int:
        """
        Returns the number of output features of the visual embedding layer.

        :return: The number of output features of the visual embedding layer.
        """
        return self.resnet50.output_dim

    def forward(self, x):
        """
        Performs a forward pass through the ResNet50 visual embedding layer.

        :param x: The input tensor.
        :return: The output tensor with the restnet50 embedded representation.
        """
        return self.resnet50(x)
