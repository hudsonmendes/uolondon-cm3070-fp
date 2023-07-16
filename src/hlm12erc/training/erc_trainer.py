# Python Built-in Modules
import json
import pathlib
import time
from typing import Optional, Tuple

# Third-Party Libraries
import transformers

# My Packages and Modules
from hlm12erc.modelling import ERCConfig, ERCModel

# Local Folders
from .erc_config_formatter import ERCConfigFormatter
from .meld_dataset import MeldDataset


class ERCTrainer:
    """
    Training class wrapper that utilises the huggingface transformers.Trainer
    class to create, train and return the best training model, based on the
    training settings and hyperparameters.

    Example:
        >>> from hlm12erc.training import ERCTrainer
        >>> from hlm12erc.training import MeldDataset
        >>> from hlm12erc.modelling import ERCConfig
        >>> config = ERCConfig()
        >>> train_dataset = MeldDataset("train")
        >>> eval_dataset = MeldDataset("dev")
        >>> trainer = ERCTrainer(config)
        >>> model = trainer.train(
        ...     data=(train_dataset, eval_dataset),
        ...     n_epochs=3)
    """

    def __init__(self, config: Optional[ERCConfig] = None):
        """
        Initialise the ERCTrainer class with the given ERCConfig object.

        :param config: ERCConfig object containing the model hyperparameters"""
        self.config = config

    def train(
        self,
        data: Tuple[MeldDataset, MeldDataset],
        n_epochs: int,
        batch_size: int,
        save_to: pathlib.Path,
    ) -> ERCModel:
        """
        Train the model using the given training and validation datasets.

        :param data: Tuple containing the training and validation datasets.
        :param n_epochs: Number of epochs to train the model for.
        :param batch_size: Batch size to use for training.
        :param save_to: Path to the directory to save the model and logs to.
        :return: ERCModel object containing the best trained model.
        """
        config = self.config or ERCConfig()
        train_dataset, eval_dataset = data
        model = ERCModel(config)
        model_name = ERCConfigFormatter(config).represent()
        workspace = save_to / model_name
        training_args = self._create_training_args(n_epochs, batch_size, model_name, workspace)
        trainer = self._create_trainer(train_dataset, eval_dataset, model, training_args)
        self._store_settings_and_hyperparams(workspace, training_args, config)
        trainer.train()
        return model

    def _create_training_args(
        self,
        n_epochs: int,
        batch_size: int,
        model_name: str,
        workspace: pathlib.Path,
    ) -> transformers.TrainingArguments:
        """
        Create the training arguments for the transformers.Trainer class.

        :param n_epochs: Number of epochs to train the model for.
        :param batch_size: Batch size to use for training.
        :param model_name: A representative model name that distiguishes its architecture.
        :param workspace: Path to the workspace to store the model and logs.
        :return: transformers.TrainingArguments object containing the training
        """
        return transformers.TrainingArguments(
            run_name=f"run-{int(time.time())}-model-{model_name}",
            do_train=True,
            do_eval=True,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            output_dir=str(workspace / "models"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            logging_dir=str(workspace / "logs"),
            logging_strategy="steps",
            logging_steps=10,
        )

    def _create_trainer(
        self,
        train_dataset: MeldDataset,
        eval_dataset: MeldDataset,
        model: ERCModel,
        training_args: transformers.TrainingArguments,
    ) -> transformers.Trainer:
        """
        Create the transformers.Trainer object to train the model.

        :param train_dataset: Training dataset to use for training.
        :param eval_dataset: Validation dataset to use for evaluation.
        :param model: ERCModel object containing the model to train.
        :param training_args: transformers.TrainingArguments object containing the training arguments.
        :return: transformers.Trainer object to train the model.
        """
        return transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    def _store_settings_and_hyperparams(
        self,
        workspace: pathlib.Path,
        training_args: transformers.TrainingArguments,
        config: ERCConfig,
    ) -> None:
        """
        Write the `training_args.json` and the `config.json` files to the workspace,
        to store both the training arguments and the model hyperparameters.

        :param workspace: Path to the workspace to store the model and logs.
        :param training_args: transformers.TrainingArguments object containing the training arguments.
        :param config: ERCConfig object containing the model hyperparameters.
        """
        with open(workspace / "training_args.json", "w") as file:
            file.write(json.dumps(training_args.__dict__))

        with open(workspace / "config.json", "w") as file:
            file.write(json.dumps(config.__dict__))
