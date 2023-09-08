# Python Built-in Modules
from typing import Any, Callable, Dict, Tuple, Union

# Third-Party Libraries
import torch
import transformers

# My Packages and Modules
from hlm12erc.modelling import ERCOutput

# Local Folders
from .erc_data_collator import ERCDataCollator


class ERCTrainerBatchJob(transformers.Trainer):
    """
    Overrides the Huggingface Trainer to add additional metrics to the training loop,
    such as accuracy, f1, precision, recall, but keeping the loss.
    """

    custom_metric_computation: Callable[[transformers.EvalPrediction], Dict[str, Any]] | None = None

    def __init__(
        self,
        compute_metrics: transformers.EvalPrediction | None = None,
        *args,
        **kwargs,
    ):
        """
        Constructs a Custom Trainer keeping the `compute_metrics` object to
        be used to calculate the metrics within the `compute_loss` function.

        :param compute_metrics: Callable object to calculate the metrics.
        """
        super(ERCTrainerBatchJob, self).__init__(compute_metrics=compute_metrics, *args, **kwargs)
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
            eval_pred = transformers.EvalPrediction(predictions=outputs.labels, label_ids=labels)
            custom_metrics = self.custom_metric_computation(eval_pred)
            metrics.update(custom_metrics)
        return metrics
