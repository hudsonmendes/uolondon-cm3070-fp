# Python Built-in Modules
import pathlib
import uuid
from abc import ABC, abstractmethod

# Third-Party Libraries
import numpy as np
import torch
import torchvision
from PIL.Image import Image
from PIL.Image import fromarray as image_from_array
from PIL.Image import open as open_image


class MeldVisualPreprocessor(ABC):
    """
    Abstract class that define the contract of visual preprocessors.
    """

    @abstractmethod
    def __call__(self, image: pathlib.Path | Image) -> Image | torch.Tensor:
        """
        when implemented, preprocesses either a filepath or an image into
        either an image or a tensor. Returning tensors should be the last
        step of the chain.

        :param image: The image to be preprocessed, either a path or an image
        :return: The preprocessed image or the final tensor
        """
        raise NotImplementedError("Not yet implemented")

    def store_into_temporary_file(self, image: Image) -> pathlib.Path:
        """
        Stores the image into a temporary file that can be used by the preproessor
        should it need the image to be stored as a file to process.

        :param image: The image to be stored
        :return: The path to the stored image
        """
        filename = f"visual_{uuid.uuid4()}.png"
        filepath = pathlib.Path(f"/tmp/hlm12erc/preprocessing/{filename}")
        image.save(filepath)
        return filepath


class MeldVisualPreprocessorFilepathToResnet50(MeldVisualPreprocessor):
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

        The preprocessing transformation routine normalises the input images using
        the meand and standard deviation calculated on the images of the training
        set, calculated as part of the mlops notebook, in the "Measures of Spread"
        section, by the function `calculate_measures_of_spread`.
        """
        self.image_preprocessor = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.Lambda(lambda x: torchvision.transforms.functional.crop(x, 0, 0, 256, 721)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.2706, 0.2010, 0.1914], std=[0.1857, 0.1608, 0.1667]),
            ]
        )

    def __call__(self, image: Image | pathlib.Path) -> Image | torch.Tensor:
        """
        Preprocesses the image by applying a series of transformations to it.

        :param x: The image to be preprocessed
        :return: The tensor fo the preprocessed image
        """
        with image if isinstance(image, Image) else open_image(image) as instance:
            return self.image_preprocessor(instance).float()


class MeldVisualPreprocessorFilepathToFaceOnlyImage(MeldVisualPreprocessor):
    """
    Preprocessor calss for the visual (image stack) files, turning them into
    tensors, for better compatibility with TPU training without affecting
    negatively CPU-based training.
    """

    def __init__(self, filepath_retinaface_resnet50: pathlib.Path, device: torch.device = None) -> None:
        """
        Creates a face detector that blacks out every pixel not found within the
        bounding box of a face.

        :param filepath_retinaface_resnet50: the path to the pretrained retinaface/resnet50 weights
        :param device: the device that will be used for inferences
        """
        # Local Folders
        from .retinaface import create_face_detector

        self.face_detector = create_face_detector(str(filepath_retinaface_resnet50), device=device)

    def __call__(self, image: Image | pathlib.Path) -> Image | torch.Tensor:
        """
        Preprocesses the image by applying a series of transformations to it.

        :param x: The image to be preprocessed
        :return: The preprocessed image
        """
        # ensure that we have a filepath to start with
        if isinstance(image, torch.Tensor):
            raise ValueError(
                """
                The input `image` is already a `torch.Tensor`, which means that the processor
                has already materialised the data into the format that the model will process
                and does not allow for preprocessing anymore."""
            )
        elif isinstance(image, Image):
            image = self.store_into_temporary_file(image)

        # detect faces in image
        faces = self.face_detector(image)

        # create the image we will manipulate
        with open_image(image) as image_instance:
            # eliminate faces that larger than 1/5 of the image
            # because it would be a face spanning over 2 frames
            # which is impossible
            faces = [face for face in faces if (face["y2"] - face["y1"]) < image_instance.height / 5]

            # black out pixels not within faces
            image_blackedout = self._black_out_non_face_pixels(image=image_instance, faces=faces)

            # returns the image instance, that can be used by the next preprocessor
            return image_blackedout

    def _black_out_non_face_pixels(self, image: Image, faces: list) -> Image:
        """
        Uses numpy to efficiently black out pixels that are not within the bounding

        :param image: The image to be manipulated
        :param faces: The list of faces detected in the image
        :return: The image with the non-face pixels blacked out
        """
        img_array = np.array(image)
        mask = np.zeros((image.height, image.width), dtype=bool)
        for face in faces:
            x1, y1, x2, y2 = face["x1"], face["y1"], face["x2"], face["y2"]
            mask[y1:y2, x1:x2] = True

        img_array[~mask] = [0, 0, 0]
        return image_from_array(img_array)
