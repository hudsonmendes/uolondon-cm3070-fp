# Python Built-in Modules
from typing import Any, Dict, Optional, Tuple, Union

# Third-Party Libraries
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction

# My Packages and Modules
from hlm12erc.modelling import ERCConfig, ERCLoss


class ERCMetricCalculator:
    """
    Responsible for calculating the metrics for an ERC prediction against
    the ground truth, including the loss if a loss function is provided.
    """

    loss_fn: Optional[ERCLoss]

    def __init__(self, config: ERCConfig) -> None:
        """
        Contructs a new ERCMetricCalculator.

        :param classifier_loss_fn: The name of the loss function to use for the classifier.
        """
        self.loss_fn = None
        if config is not None and config.classifier_loss_fn is not None:
            self.loss_fn = ERCLoss.resolve_type_from(config.classifier_loss_fn)(config)

    def __call__(self, eval_pred: EvalPrediction) -> Dict[str, Any]:
        """
        Calculates the `acc`, `f1_weighted`, `p_weighted`, `r_weighted`.
        It also produces the `loss` in case a `self.loss_fn` is provided.

        :param eval_pred: The evaluation prediction to calculate the metrics for.
        :return: A dictionary containing the metrics.
        """
        pred, labels = self._extract_pred_loss(eval_pred)
        return self._build_output(
            loss=self._determine_loss(pred=pred, labels=labels),
            y_true=labels.argmax(dim=1).cpu(),
            y_pred=pred.argmax(dim=1).cpu(),
        )

    def _extract_pred_loss(self, eval_pred) -> Tuple[torch.Tensor, torch.Tensor]:
        pred = eval_pred.predictions
        labels = eval_pred.label_ids
        pred = pred[0] if isinstance(pred, tuple) else pred
        pred = torch.from_numpy(pred) if isinstance(pred, np.ndarray) else pred
        labels = torch.from_numpy(labels) if isinstance(labels, np.ndarray) else labels
        return pred, labels

    def _determine_loss(self, pred: torch.Tensor, labels: torch.Tensor):
        return self.loss_fn(y_pred=pred, y_true=labels) if self.loss_fn is not None else None

    def _build_output(self, loss: Union[float, torch.Tensor], y_true: torch.Tensor, y_pred: torch.Tensor):
        output = {}
        if loss:
            output["loss"] = loss.item() if isinstance(loss, torch.Tensor) else loss
        output["acc"] = accuracy_score(y_true, y_pred)
        output["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
        return output
