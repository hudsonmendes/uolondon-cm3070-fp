# Python Built-in Modules
import json
import pathlib

# Third-Party Libraries
import yaml

# My Packages and Modules
from hlm12erc.modelling import ERCConfig


class ERCConfigLoader:
    """Loads the ERCConfig from a .json or a .yaml file."""

    def __init__(self, filepath: pathlib.Path):
        """
        Constructs a new ERCConfigLoader.

        :param filepath: The path to the json file.
        """
        self.filepath = filepath

    def load(self) -> ERCConfig:
        """
        Loads the ERCConfig from the json file.
        """
        config_dict = self._read_as_dict()
        return ERCConfig(**config_dict)

    def _read_as_dict(self):
        config_dict = {}
        with open(self.filepath, mode="r", encoding="utf-8") as fh:
            if self.filepath.suffix == ".json":
                config_dict = json.load(fh)
            elif self.filepath.suffix == ".yaml":
                config_dict = yaml.safe_load(fh)
        return config_dict
