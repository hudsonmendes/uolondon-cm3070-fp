# Python Built-in Modules
from abc import ABC
from dataclasses import dataclass
from typing import List
from wave import Wave_read as Wave

# Third-Party Libraries
from PIL.Image import Image


@dataclass(frozen=True)
class MeldDialogueEntry(ABC):
    """
    Represents each record of the MELD datset.
    """

    speaker: str
    utterance: str

    def to_utterance_prompt(self) -> str:
        """
        Returns the text prompt for the sample, which is the concatenation of
        the previous dialogue and the current utterance.

        :return: The text prompt for the sample
        """
        return f"{self.speaker} said:###\n{self.utterance}\n###"


@dataclass(frozen=True)
class MeldRecord(MeldDialogueEntry):
    """
    Represents each record of the MELD datset.
    """

    audio: Wave
    visual: Image
    previous_dialogue: List[MeldDialogueEntry]
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
