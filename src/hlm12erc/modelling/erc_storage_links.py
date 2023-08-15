# Python Built-in Modules
import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class ERCStorageLinks:
    """
    Represents a package of ERCPaths, containing path pointers to the
    model folder, the configuration file and the training arguments file.
    """

    pth: pathlib.Path
    config: pathlib.Path
    training_args: pathlib.Path
