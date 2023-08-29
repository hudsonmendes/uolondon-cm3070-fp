# Python Built-in Modules
import pathlib
import uuid
from abc import ABC, abstractmethod

# Third-Party Libraries
import torch
import torchvision
from PIL.Image import Image
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

    image_preprocessor: torchvision.transforms.Compose

    def __init__(self, filepath_retinaface_resnet50: pathlib.Path) -> None:
        """
        Creates a face detector that blacks out every pixel not found within the
        bounding box of a face.
        """
        # Local Folders
        from .retinaface import create_face_detector

        self.face_detector = create_face_detector(filepath_retinaface_resnet50)

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

        # black out pixels not within faces
        image_instance = open_image(image)
        for x in range(image_instance.width):
            for y in range(image_instance.height):
                pixel = (x, y)
                if not self._is_pixel_within_face_bbox(pixel, faces):
                    image_instance.putpixel(pixel, (0, 0, 0))

        # returns the image instance, that can be used by the next preprocessor
        return image_instance

    def _is_pixel_within_face_bbox(self, pixel: tuple, faces: list) -> bool:
        """
        Retuns True whenever the pixel is found within the bounding box of
        a face, and false if not within any.

        :param pixel: tuple with x, y coordinates
        :param faces: the list of dictionaries describing bounding boxes of faces.
        """
        x, y = pixel
        for face in faces:
            x1, y1, w, h = face["x"], face["y"], face["width"], face["height"]
            x2, y2 = x1 + w, y1 + h
            if x1 <= x <= x2 and y1 <= y <= y2:
                return True
        return False
