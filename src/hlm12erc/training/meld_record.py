# Python Built-in Modules
from dataclasses import dataclass

# Third-Party Libraries
import torch


@dataclass(frozen=True)
class MeldRecord:
    """
    Represents each record of the MELD datset.
    """

    audio: torch.Tensor
    visual: torch.Tensor
    text: str
    label: str
