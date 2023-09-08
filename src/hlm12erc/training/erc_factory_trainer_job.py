# Third-Party Libraries
import transformers

# My Packages and Modules
from hlm12erc.modelling import ERCConfig, ERCLabelEncoder, ERCLossFunctions, ERCModel

# Local Folders
from .erc_data_collator import ERCDataCollator
from .erc_metric_calculator import ERCMetricCalculator
from .erc_trainer_job_batch import ERCTrainerBatchJob
from .erc_trainer_job_triplet import ERCTrainerTripletJob
from .meld_dataset import MeldDataset


class ERCTrainerJobFactory:
    """
    Factory class to create the transformers.Trainer object to train the model,
    based on the configuration, for a given model, and with the datasets provided.
    """

    def __init__(self, config: ERCConfig):
        """
        Initialise the ERCTrainerJobFactory class with the given ERCConfig object.

        :param config: ERCConfig object containing the model hyperparameters
        """
        self.config = config

    def create(
        self,
        model: ERCModel,
        train_dataset: MeldDataset,
        eval_dataset: MeldDataset,
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
        if not self._should_use_triplet_loss():
            return ERCTrainerBatchJob(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=ERCDataCollator(config=self.config, label_encoder=label_encoder),
                compute_metrics=ERCMetricCalculator(config=self.config),
                callbacks=self._create_callbacks(),
            )
        else:
            return ERCTrainerTripletJob(
                model=model,
                config=self.config,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=ERCDataCollator(config=self.config, label_encoder=label_encoder),
                compute_metrics=ERCMetricCalculator(config=self.config),
                callbacks=self._create_callbacks(),
            )

    def _create_callbacks(self):
        callbacks = []
        if (
            self.config.classifier_early_stopping_patience is not None
            and self.config.classifier_early_stopping_patience > 0
        ):
            callbacks.append(
                transformers.EarlyStoppingCallback(
                    early_stopping_patience=self.config.classifier_early_stopping_patience,
                    early_stopping_threshold=0.001,
                )
            )
        return callbacks

    def _should_use_triplet_loss(self):
        triplet_suffix = "+" + ERCLossFunctions.TRIPLET
        return self.config.classifier_loss_fn.endswith(triplet_suffix)
