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


class ERCPath:
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
        model.load_state_dict(torch.load(str(self.dir / ERCPath.MODEL_FILENAME)))
        return model

    def save(self, model: ERCModel, ta: TrainingArguments) -> None:
        """
        Saves a model into the folder, saving its configuration into
        `config.json`, its training arguments into `training_args.json`
        and its weights into `model.pt`.

        :param model: Model to be saved.
        :param ta: Training arguments used to train the model.
        """

        self.dir.mkdir(parents=True, exist_ok=True)
        self._write_training_args(workspace=self.dir, ta=ta)
        self._write_config(workspace=self.dir, config=model.config)
        torch.save(model.state_dict(), str(self.dir / ERCPath.MODEL_FILENAME))

    def _write_training_args(self, workspace: pathlib.Path, ta: TrainingArguments) -> None:
        """
        Write the `training_args.json` file to the workspace, to store the training arguments.

        :param workspace: Path to the workspace to store the model and logs.
        """
        with open(workspace / "training_args.json", "w") as file:
            file.write(json.dumps(ta.to_dict(), indent=4))

    def _write_config(self, workspace: pathlib.Path, config: ERCConfig) -> None:
        """
        Write the `config.json` file to the workspace, to store the model hyperparameters.

        :param workspace: Path to the workspace to store the model and logs.
        :param config: ERCConfig object containing the model hyperparameters.
        """
        with open(workspace / "config.json", "w") as file:
            doc = config.__dict__
            doc["feedforward_layers"] = [layer.__dict__ for layer in doc["feedforward_layers"]]
            file.write(json.dumps(doc, indent=4))
