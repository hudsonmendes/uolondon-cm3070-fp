import pathlib
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MeldDataRecord:
    audio: pathlib.Path
    video: pathlib.Path
    dialogue: List[str]
    utterance: str
