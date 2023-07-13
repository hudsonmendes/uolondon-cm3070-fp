from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KaggleDataset:
    owner: str
    name: str
    subdir: Optional[str] = None
