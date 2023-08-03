# Python Built-in Modules
from abc import ABC, abstractmethod
from typing import Optional, Type

# Third-Party Libraries
import torch

# Local Folders
from .erc_config import ERCLossFunctions


class ERCLoss(ABC):
    """
    Defines the contract of loss functions for ERC models.

    Example:
        >>> loss = ERCLoss.resolve_type_from("cce")()
        >>> loss(y_true=y_true, y_pred=y_pred)
    """

    @abstractmethod
    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
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
        else:
            raise ValueError(f"Unknown ERC Loss type {expression}")


class CategoricalCrossEntropyLoss(ERCLoss):
    """
    Categorical Cross Entropy Loss function for ERC models.
    """

    def __init__(self):
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
