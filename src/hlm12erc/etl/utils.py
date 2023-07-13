# Python Built-in Modules
import pathlib
from typing import Union


def ensure_path(path: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Ensures that the path is a pathlib.Path.
    :param path: The path.
    :return: The path as a pathlib.Path.
    """
    if isinstance(path, str):
        path = pathlib.Path(path)
    return path
