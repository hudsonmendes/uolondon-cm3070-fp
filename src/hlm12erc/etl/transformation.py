import csv
import itertools
import pathlib
from typing import Union


class RawTo1NFTransformer:
    """
    Transform a raw dataset into a 1NF dataset that can be
    more easily stored and consumed during training.
    """

    src: pathlib.Path

    def __init__(self, src: Union[str, pathlib.Path]) -> None:
        """
        Create a new transformer that transforms the raw dataset
        from the source file.
        :param src: The source file to transform.
        """
        self.src = pathlib.Path(src)

    def transform(self, dest: Union[str, pathlib.Path]) -> None:
        """
        Transform the raw dataset from the source file and save it
        to the destination file.
        :param dest: The destination file to save the transformed dataset to.
        """
        raise NotImplementedError("Not yet implemented")
