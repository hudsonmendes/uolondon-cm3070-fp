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

    def to_dialogue_prompt(self) -> str:
        """
        Returns the text prompt for the sample, which is the concatenation of
        the previous dialogue and the current utterance.

        :return: The text prompt for the sample
        """
        sep = "\n\n\n"
        before = sep.join([entry.to_utterance_prompt() for entry in self.previous_dialogue])
        current = sep.join([before, self.to_utterance_prompt()])
        return sep.join([before, current])
