# Python Built-in Modules
import pathlib
import time

# Third-Party Libraries
import transformers

# My Packages and Modules
from hlm12erc.modelling import ERCConfig

# Local Folders
from .erc_data_collator import ERCDataCollator


class ERCTrainerJobTrainingArgsFactory:
    """
    Factory class to create the transformers.TrainingArguments object to train the model,
    based on the configuration and training settings provided.
    """

    def __init__(self, config: ERCConfig):
        """
        Initialise the ERCTrainerJobTrainingArgsFactory class with the given ERCConfig object.

        :param config: ERCConfig object containing the model hyperparameters
        """
        self.config = config

    def create(
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
            learning_rate=self.config.classifier_learning_rate,
            weight_decay=self.config.classifier_weight_decay,
            warmup_steps=self.config.classifier_warmup_steps,
            metric_for_best_model=self.config.classifier_metric_for_best_model,
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
