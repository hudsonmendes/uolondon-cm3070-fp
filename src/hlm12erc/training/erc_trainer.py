# Python Built-in Modules
import json
import logging
import pathlib
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Third-Party Libraries
import torch
import transformers

# My Packages and Modules
from hlm12erc.modelling import ERCConfig, ERCLabelEncoder, ERCModel, ERCOutput

# Local Folders
from .erc_config_formatter import ERCConfigFormatter
from .erc_data_collator import ERCDataCollator
from .erc_metric_calculator import ERCMetricCalculator
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
    ) -> Tuple[str, ERCModel]:
        """
        Train the model using the given training and validation datasets.

        :param data: Tuple containing the training and validation datasets.
        :param n_epochs: Number of epochs to train the model for.
        :param batch_size: Batch size to use for training.
        :param save_to: Path to the directory to save the model and logs to.
        :return: The name of the model and the ERCModel object containing the best trained model.
        """
        logger.info("Training the model...")
        train_dataset, eval_dataset = data
        logger.info("Training & Validation datasets unpacked")
        label_encoder = ERCLabelEncoder(classes=train_dataset.classes)
        logger.info(f"Label Encoder loaded with classes: {', '.join(label_encoder.classes)}")
        config = self.config or ERCConfig()
        logger.info(f"Training with config {'passed to trainer' if self.config else 'default'}")
        model = ERCModel(config=config, label_encoder=label_encoder)
        model_name = ERCConfigFormatter(config).represent()
        logger.info(f"Model created for training, with identifier {model_name}")
        workspace = save_to / model_name
        logger.info(f"Training workspace set to: {workspace}")
        training_args = self._create_training_args(n_epochs, batch_size, model_name, workspace)
        logger.info(f"TrainingArgs created with {n_epochs} epochs and batch size {batch_size}")
        trainer = self._create_trainer(train_dataset, eval_dataset, model, training_args, label_encoder)
        logger.info(f"Trainer instantiated with samples train={len(train_dataset)}, valid={len(eval_dataset)}")
        self._store_settings_and_hyperparams(workspace, training_args, config)
        logger.info("Training settings and hyperparameters stored in workspace")
        logger.info("Training starting, don't wait standing up...")
        trainer.train()
        logger.info("Training complete.")
        return model_name, model

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
            label_names=[ERCDataCollator.LABEL_NAME],
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
            disable_tqdm=False,
            report_to=["wandb"],
        )

    def _create_trainer(
        self,
        train_dataset: MeldDataset,
        eval_dataset: MeldDataset,
        model: ERCModel,
        training_args: transformers.TrainingArguments,
        label_encoder: ERCLabelEncoder,
    ) -> transformers.Trainer:
        """
        Create the transformers.Trainer object to train the model.

        :param train_dataset: Training dataset to use for training.
        :param eval_dataset: Validation dataset to use for evaluation.
        :param model: ERCModel object containing the model to train.
        :param training_args: transformers.TrainingArguments object containing the training arguments.
        :param label_encoder: ERCLabelEncoder object containing the label encoder to use for training.
        :return: transformers.Trainer object to train the model.
        """
        classifier_loss_fn = self.config.classifier_loss_fn if self.config else None
        return _ERCHuggingfaceCustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=ERCDataCollator(label_encoder=label_encoder),
            compute_metrics=ERCMetricCalculator(classifier_loss_fn=classifier_loss_fn),
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
        self._write_training_args(workspace, training_args)
        self._write_config(workspace, config)

    def _write_training_args(self, workspace: pathlib.Path, training_args: transformers.TrainingArguments) -> None:
        """
        Write the `training_args.json` file to the workspace, to store the training arguments.

        :param workspace: Path to the workspace to store the model and logs.
        """
        with open(workspace / "training_args.json", "w") as file:
            file.write(json.dumps(training_args.to_dict(), indent=4))

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


class _ERCHuggingfaceCustomTrainer(transformers.Trainer):
    """
    Overrides the Huggingface Trainer to add additional metrics to the training loop,
    such as accuracy, f1, precision, recall, but keeping the loss.
    """

    custom_metric_computation: Optional[Callable[[transformers.EvalPrediction], Dict[str, Any]]] = None

    def __init__(
        self,
        compute_metrics: Optional[transformers.EvalPrediction] = None,
        *args,
        **kwargs,
    ):
        """
        Constructs a Custom Trainer keeping the `compute_metrics` object to
        be used to calculate the metrics within the `compute_loss` function.

        :param compute_metrics: Callable object to calculate the metrics.
        """
        super(_ERCHuggingfaceCustomTrainer, self).__init__(compute_metrics=compute_metrics, *args, **kwargs)
        self.custom_metric_computation = compute_metrics

    def compute_loss(self, model, inputs, return_outputs=False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Uses the model to calculate outputs over the inputs, and then calculates
        both the loss, accuracy, f1, precision and recall.

        :param model: ERCModel object containing the model to train.
        :param inputs: Dictionary containing the inputs to the model.
        :param return_outputs: Whether to return the outputs of the model.
        :return: Tuple containing the loss and the outputs of the model.
        """
        outputs = model(**inputs)
        loss = outputs.loss
        labels = inputs.get(ERCDataCollator.LABEL_NAME)
        if labels is not None:
            metrics = self._compute_metrics(labels, outputs, loss=loss.item())
            self.log(metrics)
        return (loss, outputs) if return_outputs else loss

    def _compute_metrics(self, labels: torch.Tensor, outputs: ERCOutput, loss: float) -> Dict[str, Any]:
        """
        Runs the `custom_metric_computation` procedure to calculate the metrics
        that we want to track, and then updates the metrics dictionary with the
        loss and the custom metrics.

        :param labels: Labels of the inputs.
        :param outputs: Outputs of the model.
        :param loss: Loss of the model, float format (tensor.item())
        :return: Dictionary containing the metrics.
        """
        metrics = dict(loss=loss)
        if self.custom_metric_computation is not None:
            eval_pred = transformers.EvalPrediction(predictions=labels, label_ids=outputs.labels)
            custom_metrics = self.custom_metric_computation(eval_pred)
            metrics.update(custom_metrics)
        return metrics
