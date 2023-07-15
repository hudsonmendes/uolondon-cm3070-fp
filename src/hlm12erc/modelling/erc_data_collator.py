# Third-Party Libraries
import torch
import torchvision
from PIL.Image import Image


class ERCDataCollator:
    """
    Collates the data from the ERC dataset into a format that can be used and
    batched by the model, with multiple records turned into matrices.
    """

    @staticmethod
    def transform_image(image: Image) -> torch.Tensor:
        """
        Resizes, CenterCrops and Converts the image to a tensor.

        :param image: One image to transform.
        :return: The tensor representing one image
        """
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
            ]
        )(image).unsqueeze(0)[0]
