# Third-Party Libraries
import torch
import torchvision
from PIL.Image import Image


class MeldVisualPreprocessor:
    """
    Preprocessor calss for the visual (image stack) files, turning them into
    tensors, for better compatibility with TPU training without affecting
    negatively CPU-based training.
    """

    image_preprocessor: torchvision.transforms.Compose

    def __init__(self) -> None:
        """
        Creates a new instance of the MeldVideoPreprocessor class with the
        default image preprocessor (using torchvision.transforms)
        """
        self.image_preprocessor = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.Lambda(lambda x: torchvision.transforms.functional.crop(x, 0, 0, 256, 721)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.2706, 0.2010, 0.1914], std=[0.1857, 0.1608, 0.1667]),
            ]
        )

    def __call__(self, x: Image) -> torch.Tensor:
        """
        Preprocesses the image by applying a series of transformations to it.

        :param x: The image to be preprocessed
        :return: The preprocessed image
        """
        return self.image_preprocessor(x).float()
