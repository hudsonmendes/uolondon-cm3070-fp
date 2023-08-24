# Python Built-in Modules
from abc import abstractmethod
from typing import Callable, Optional, Type

# Third-Party Libraries
import torch
from torch.nn.functional import normalize as l2_norm
from transformers import AutoFeatureExtractor, AutoModelForObjectDetection

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig

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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        When implemented, this method should receive a list of images and
        return a matrix of tensors (batch_size, out_features).

        :param x: stacked vectors representing images
        :return: matrix of tensors (batch_size, out_features)
        """
        raise NotImplementedError("The method 'forward' must be implemented.")

    @staticmethod
    def resolve_type_from(
        expression: str,
    ) -> Type["ERCVisualEmbeddings"] | Callable[[ERCConfig], Optional["ERCVisualEmbeddings"]]:
        if expression == ERCVisualEmbeddingType.RESNET50:
            return ERCResNet50VisualEmbeddings
        elif expression == ERCVisualEmbeddingType.DTR_RESNET50:
            return ERCFaceOnlyResNet50VisualEmbeddings
        elif expression == ERCVisualEmbeddingType.NONE:
            return lambda _: None
        raise ValueError(f"The visual embedding '{expression}' is not supported.")


class ERCResNet50VisualEmbeddings(ERCVisualEmbeddings):
    """
    ERCResNet50VisualEmbeddings is a class that implements the visual embedding
    layer using ResNet50 and simply returning the output of the final layer.

    References:
    >>> "Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition.
    ... In Proceedings of the IEEE conference on computer vision and pattern recognition, 770–778."

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

        The resnet50 model is loaded from the torchvision library, and the final
        fully connected layer is replaced by an identity layer, so that the output
        of the model is the output of the final convolutional layer.

        :param config: The configuration for the ERC model.
        """
        super().__init__(config=config)
        self.resnet50 = torch.hub.load("pytorch/vision:v0.6.0", "resnet50", pretrained=True)
        self.resnet50.eval()
        self.resnet50.fc = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the ResNet50 visual embedding layer.

        :param x: stacked vectors representing images
        :return: matrix of tensors (batch_size, out_features)
        """
        y = self.resnet50(x)
        y = l2_norm(y, p=2, dim=1)
        return y

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the embedding transformation.
        """
        return 2048


class ERCFaceOnlyResNet50VisualEmbeddings(ERCResNet50VisualEmbeddings):
    """
    ERCTinaFaceResNet50VisualEmbeddings is a class that implements the visual
    embeddings layer using the ResNet50 model on top of the images cropped
    to only the faces of the actors, rather than the entire scene.

    Reference (DTR, Face Detection):
    >>> Yanjia Zhu, Hongxiang Cai, Shuhan Zhang, Chenhao Wang, and Yichao Xiong. 2021.
    ... TinaFace: Strong but Simple Baseline for Face Detection.

    Reference (ResNet50):
    >>> "Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition.
    ... In Proceedings of the IEEE conference on computer vision and pattern recognition, 770–778."

    Example:
        >>> from hlm12erc.modelling import ERCConfig, ERCVisualEmbeddingType
        >>> from hlm12erc.modelling.erc_emb_visual import ERCVisualEmbeddings
        >>> config = ERCConfig()
        >>> ERCVisualEmbeddings.resolve_type_from(ERCVisualEmbeddingType.DTR_RESNET50)(config)
    """

    def __init__(self, config: ERCConfig):
        """
        Creates a new instance of ERCTinaFaceResNet50VisualEmbeddings, based off
        the ERCResNet50VisualEmbeddings. The main difference is a pre-processing
        step through which the faces are detected from the image. Everything else
        is masked (black pixels) before being passed into the super().forward(cropped_images).

        :param config: The configuration for the ERC model.
        """
        super().__init__(config=config)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("aditmohan96/detr-finetuned-face")
        self.face_detector = AutoModelForObjectDetection.from_pretrained("aditmohan96/detr-finetuned-face")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the ResNet50 visual embedding layer,
        but brefore masks the image to only the faces of the actors.

        :param x: stacked vectors representing images
        :return: matrix of tensors (batch_size, out_features)
        """
        y = self._mask_out_non_faces_from(x)
        y = self.forward(y)
        return y

    def _mask_out_non_faces_from(self, x: torch.Tensor) -> torch.Tensor:
        """
        Masks out (replace by a black pixel) anything that does not fall within
        the bounding box of a face detected by the face detector.

        :param x: stacked vectors representing images, shape (batch_size, 3, 256, 721)
        :return: matrix of tensors (batch_size, 3, 256, 721)
        """
        f = self.feature_extractor(x)
        m = self.face_detector(f, return_tensors="pt")
        assert m is not None
        # TODO: mask out everything not a face
        return x
