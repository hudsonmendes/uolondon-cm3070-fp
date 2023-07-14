# Third-Party Libraries
import torch

# Local Folders
from .erc_config import ERCConfig, ERCVisualEmbeddingType
from .erc_emb import ERCEmbeddings


class ERCVisualEmbeddings(ERCEmbeddings):
    """
    ERCVisualEmbeddings is an abstract class that defines the interface for visual embedding layers.

    Examples:
        >>> from hlm12erc.modelling.erc_emb_visual import ERCVisualEmbeddingType, ERCVisualEmbeddings
        >>> ERCVisualEmbeddings.resolve_type_from(ERCVisualEmbeddingType.RESNET_50)
    """

    @staticmethod
    def resolve_type_from(expression: str) -> type["ERCVisualEmbeddings"]:
        if expression == ERCVisualEmbeddingType.RESNET_50:
            return ERCResNet50VisualEmbeddings
        raise ValueError(f"The visual embedding '{expression}' is not supported.")


class ERCResNet50VisualEmbeddings(ERCVisualEmbeddings):
    """
    ERCResNet50VisualEmbeddings is a class that implements the visual embedding
    layer using ResNet50 and simply returning the output of the final layer.

    Examples:
        >>> from hlm12erc.modelling.erc_config import ERCConfig
        >>> from hlm12erc.modelling.erc_emb_visual import ERCResNet50VisualEmbeddings
        >>> config = ERCConfig()
        >>> embeddings = ERCResNet50VisualEmbeddings(config)
        >>> embeddings(torch.randn(1, 3, 224, 224))
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
        return self.resnet50.out_features

    def forward(self, x):
        """
        Performs a forward pass through the ResNet50 visual embedding layer.

        :param x: The input tensor.
        :return: The output tensor.
        """
        return self.resnet50(x)
