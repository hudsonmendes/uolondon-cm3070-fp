# Python Built-in Modules
from dataclasses import dataclass

# Third-Party Libraries
import torch


@dataclass(frozen=True)
class MeldRecord:
    """
    Represents each record of the MELD datset.
    """

    text: str | None
    visual: torch.Tensor | None
    audio: torch.Tensor | None
    label: str
