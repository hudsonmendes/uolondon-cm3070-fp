# Python Built-in Modules
from abc import ABC, abstractmethod
from typing import Optional, Type

# Third-Party Libraries
import torch


class ERCLoss(ABC):
    """
    Defines the contract of loss functions for ERC models.

    Example:
        >>> loss = ERCLoss.resolve_type_from("cross_entropy")()
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
    def resolve_type_from(expression: str) -> Type[torch.nn.Module]:
        """
        Resolve a ERC Loss class from a string expression
        """
        if expression == "cross_entropy":
            return torch.nn.CrossEntropyLoss
        else:
            raise ValueError(f"Unknown ERC Loss type {expression}")
