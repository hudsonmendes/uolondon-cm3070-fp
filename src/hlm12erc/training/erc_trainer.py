# Python Built-in Modules
import logging
import pathlib
import time
from typing import Optional, Tuple

# Third-Party Libraries
import torch
import wandb

# My Packages and Modules
from hlm12erc.modelling import ERCConfig, ERCLabelEncoder, ERCModel, ERCStorage, ERCStorageLinks

# Local Folders
from .erc_config_formatter import ERCConfigFormatter
from .erc_factory_trainer_job import ERCTrainerJobFactory
from .erc_factory_trainer_ta import ERCTrainerJobTrainingArgsFactory
from .meld_dataset import MeldDataset

logger = logging.getLogger(__name__)


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
        device: Optional[torch.device] = None,
    ) -> Tuple[str, ERCModel]:
        """
        Train the model using the given training and validation datasets.

        :param data: Tuple containing the training and validation datasets.
        :param n_epochs: Number of epochs to train the model for.
        :param batch_size: Batch size to use for training.
        :param save_to: Path to the directory to save the model and logs to.
        :param device: device to use for data collator and training.
        :return: The name of the model and the ERCModel object containing the best trained model.
        """
        logger.info("Training the model...")
        train_dataset, eval_dataset = data
        logger.info("Training & Validation datasets unpacked")
        config = self.config or ERCConfig(classifier_classes=train_dataset.classes)
        logger.info(f"Training with config {'passed to trainer' if self.config else 'default'}")
        label_encoder = ERCLabelEncoder(classes=config.classifier_classes)
        logger.info(f"Label Encoder loaded with classes: {', '.join(label_encoder.classes)}")
        model_name = ERCConfigFormatter(config).represent()
        logger.info(f"Model identifier {model_name}")
        model = ERCModel(config=config, label_encoder=label_encoder)
        if device is not None:
            model.to(device)
        logger.info(f"Model created in device {model.device}")
        workspace = save_to / model_name
        logger.info(f"Training workspace set to: {workspace}")
        training_args = ERCTrainerJobTrainingArgsFactory(config).create(
            n_epochs=n_epochs,
            batch_size=batch_size,
            model_name=model_name,
            workspace=workspace,
        )
        logger.info(f"TrainingArgs created with {n_epochs} epochs and batch size {batch_size}")
        trainer = ERCTrainerJobFactory(config).create(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model=model,
            training_args=training_args,
            label_encoder=label_encoder,
        )
        logger.info(f"Trainer, train={len(train_dataset)}, valid={len(eval_dataset)}, device={model.device}")
        logger.info("Training starting now, don't wait standing up...")
        trainer.train()
        logger.info("Training complete, saving model...")
        links = ERCStorage(workspace).save(model=model, ta=training_args)
        logger.info(f"Model saved into disk {links.pth}")
        if wandb.run is not None:
            logger.info("Uploading Metrics & Artifacts to W&B, ...")
            self._wanb_upload_artifact(model_name=model_name, links=links)
            wandb.finish()
            logger.info("W&B run marked as completed.")
        return model_name, model

    def _wanb_upload_artifact(self, model_name: str, links: ERCStorageLinks) -> None:
        artifact = wandb.Artifact(model_name, type="model")
        artifact.add_file(str(links.pth))
        artifact.add_file(str(links.training_args))
        artifact.add_file(str(links.config))
        artifact.save()
