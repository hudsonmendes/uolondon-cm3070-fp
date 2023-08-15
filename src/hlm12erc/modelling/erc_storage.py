# Python Built-in Modules
import json
import pathlib

# Third-Party Libraries
import torch
from transformers import TrainingArguments

# My Packages and Modules
from hlm12erc.modelling import ERCConfig, ERCLabelEncoder, ERCModel

# Local Folders
from .erc_config_loader import ERCConfigLoader
from .erc_storage_links import ERCStorageLinks


class ERCStorage:
    """
    Represents a folder in the disk from where a model is loaded and
    into where it is stored with its necessary configuration.
    """

    MODEL_FILENAME: str = "model.pt"

    def __init__(self, dir: pathlib.Path) -> None:
        """
        Constructs a new ERCPath object.

        :param dir: Path to the folder where the model is stored.
        """
        self.dir = dir

    def load(self) -> ERCModel:
        """
        Loads a model from the folder, using the `config.json` stored into it,
        instantiating the label encoder and loading the model weights.
        """
        config = ERCConfigLoader(self.dir / "config.json").load()
        label_encoder = ERCLabelEncoder(classes=config.classifier_classes)
        model = ERCModel(config, label_encoder=label_encoder)
        model.load_state_dict(torch.load(str(self.dir / ERCStorage.MODEL_FILENAME)))
        return model

    def save(self, model: ERCModel, ta: TrainingArguments) -> ERCStorageLinks:
        """
        Saves a model into the folder, saving its configuration into
        `config.json`, its training arguments into `training_args.json`
        and its weights into `model.pt`.

        :param model: Model to be saved.
        :param ta: Training arguments used to train the model.
        :return: ERCStorageLinks object containing the paths to the saved files.
        """

        self.dir.mkdir(parents=True, exist_ok=True)
        filepath_ta = self.dir / "training_args.json"
        filepath_config = self.dir / "config.json"
        filepath_model = self.dir / ERCStorage.MODEL_FILENAME
        self._write_training_args(filepath_ta, ta=ta)
        self._write_config(filepath_config, config=model.config)
        torch.save(model.state_dict(), str(filepath_model))
        return ERCStorageLinks(pth=filepath_model, config=filepath_config, training_args=filepath_ta)

    def _write_training_args(self, filepath: pathlib.Path, ta: TrainingArguments) -> None:
        """
        Write the `training_args.json` file to the workspace, to store the training arguments.

        :param filepath: Path to the workspace to store the model and logs.
        """
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(json.dumps(ta.to_dict(), indent=4))

    def _write_config(self, filepath: pathlib.Path, config: ERCConfig) -> None:
        """
        Write the `config.json` file to the workspace, to store the model hyperparameters.

        :param workspace: Path to the workspace to store the model and logs.
        :param config: ERCConfig object containing the model hyperparameters.
        """
        with open(filepath, "w", encoding="utf-8") as file:
            doc = config.__dict__
            doc["feedforward_layers"] = [layer.__dict__ for layer in doc["feedforward_layers"]]
            file.write(json.dumps(doc, indent=4))
