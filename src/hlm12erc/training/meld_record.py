# Python Built-in Modules
from dataclasses import dataclass
from typing import List
from wave import Wave_read as Wave

# Third-Party Libraries
from PIL.Image import Image


@dataclass(frozen=True)
class MeldRecord:
    audio: Wave
    visual: Image
    dialogue: List[str]
    utterance: str
