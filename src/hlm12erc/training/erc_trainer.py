# Python Built-in Modules
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
from .erc_path import ERCPath
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
        training_args = self._create_training_args(
            n_epochs=n_epochs,
            batch_size=batch_size,
            model_name=model_name,
            workspace=workspace,
        )
        logger.info(f"TrainingArgs created with {n_epochs} epochs and batch size {batch_size}")
        trainer = self._create_trainer(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model=model,
            training_args=training_args,
            label_encoder=label_encoder,
            config=config,
        )
        logger.info(f"Trainer, train={len(train_dataset)}, valid={len(eval_dataset)}, device={model.device}")
        logger.info("Training starting now, don't wait standing up...")
        trainer.train()
        logger.info("Training complete, saving model...")
        ERCPath(workspace).save(model=model, ta=training_args)
        logger.info("Model saved, thanks for waiting!")
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
        :param workspace: Path to the workspace to store the model and logs.\
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
        config: ERCConfig,
    ) -> transformers.Trainer:
        """
        Create the transformers.Trainer object to train the model.

        :param train_dataset: Training dataset to use for training.
        :param eval_dataset: Validation dataset to use for evaluation.
        :param model: ERCModel object containing the model to train.
        :param training_args: transformers.TrainingArguments object containing the training arguments.
        :param label_encoder: ERCLabelEncoder object containing the label encoder to use for training.
        :param config: ERCConfig object containing the model configuration.
        :return: transformers.Trainer object to train the model.
        """
        classifier_loss_fn = self.config.classifier_loss_fn if self.config else None
        return _ERCHuggingfaceCustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=ERCDataCollator(config=config, label_encoder=label_encoder),
            compute_metrics=ERCMetricCalculator(classifier_loss_fn=classifier_loss_fn),
        )


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
