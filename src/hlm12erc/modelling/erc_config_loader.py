# Python Built-in Modules
import json
import pathlib

# Third-Party Libraries
import yaml

# My Packages and Modules
from hlm12erc.modelling import ERCConfig, ERCConfigFeedForwardLayer


class ERCConfigLoader:
    """Loads the ERCConfig from a .json or a .yaml file."""

    def __init__(self, filepath: pathlib.Path):
        """
        Constructs a new ERCConfigLoader.

        :param filepath: The path to the json file.
        """
        assert isinstance(filepath, pathlib.Path)
        self.filepath = filepath

    def load(self) -> ERCConfig:
        """
        Loads the ERCConfig from the json file.
        """
        raw = self._read_as_dict()
        if "feedforward_layers" in raw and raw["feedforward_layers"]:
            raw["feedforward_layers"] = self._convert_ff_layers_dict(raw["feedforward_layers"])
        if "visual_in_features" in raw and raw["visual_in_features"]:
            raw["visual_in_features"] = tuple(raw["visual_in_features"])
        return ERCConfig(**raw)

    def _read_as_dict(self):
        config_dict = {}
        with open(self.filepath, mode="r", encoding="utf-8") as fh:
            if self.filepath.suffix == ".json":
                config_dict = json.load(fh)
            elif self.filepath.suffix in [".yml", ".yaml"]:
                config_dict = yaml.safe_load(fh)
        return config_dict

    def _convert_ff_layers_dict(self, ff_layers_dict):
        ff_layers = []
        for layer in ff_layers_dict:
            ff_layers.append(ERCConfigFeedForwardLayer(**layer))
        return ff_layers
