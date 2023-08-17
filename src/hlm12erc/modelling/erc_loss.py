# Python Built-in Modules
from abc import ABC, abstractmethod
from typing import Optional, Type

# Third-Party Libraries
import torch
import torch.nn.functional as F

# Local Folders
from .erc_config import ERCConfig, ERCLossFunctions


class ERCLoss(ABC):
    """
    Defines the contract of loss functions for ERC models.

    Example:
        >>> loss = ERCLoss.resolve_type_from("cce")()
        >>> loss(y_true=y_true, y_pred=y_pred)
    """

    config: ERCConfig

    def __init__(self, config: ERCConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        When implemented, this method should calculate and return the loss
        given the predicted and true labels.
        """
        raise NotImplementedError("ERCLoss is an abstract class.")

    @staticmethod
    def resolve_type_from(expression: str) -> Type["ERCLoss"]:
        """
        Resolve a ERC Loss class from a string expression
        """
        if expression == ERCLossFunctions.CATEGORICAL_CROSS_ENTROPY:
            return CategoricalCrossEntropyLoss
        elif expression == ERCLossFunctions.DICE_COEFFICIENT:
            return DiceCoefficientLoss
        else:
            raise ValueError(f"Unknown ERC Loss type {expression}")


class CategoricalCrossEntropyLoss(ERCLoss):
    """
    Categorical Cross Entropy Loss function for ERC models.
    """

    def __init__(self, config: ERCConfig):
        super().__init__(config=None)
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate and return the loss given the predicted and true labels.
        """
        return self.loss(y_pred, y_true)


class DiceCoefficientLoss(ERCLoss):
    """
    Dice Coefficient Loss function for ERC models.
    """

    def __init__(self, config: ERCConfig):
        super().__init__(config=config)
        self.epsilon = config.classifier_epsilon

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate and return the loss given the predicted and true labels
        using the Dice Loss function, which is defined as mean of the Dice
        coefficient across all classes, and derived from the equation:
        >>> 1 - (2 * TP) / (2 * TP + FP + FN)

        :param y_pred: Predicted labels, already converted to a batch of softmax probability distributions
        :param y_true: True labels
        :return: Loss value
        """
        # Compute TP, FP, and FN for each class
        assert y_true is not None
        TP = (y_pred * y_true).sum(dim=0)
        FP = (y_pred * (1 - y_true)).sum(dim=0)
        FN = ((1 - y_pred) * y_true).sum(dim=0)

        # Compute Dice coefficient for each class
        dice_class = (2 * TP) / (2 * TP + FP + FN + self.epsilon)

        # Average Dice coefficient across all classes and compute the loss
        return 1 - dice_class.mean()
