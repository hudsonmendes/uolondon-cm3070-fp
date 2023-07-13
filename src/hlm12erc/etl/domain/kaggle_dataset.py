from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class KaggleDataset:
    """
    Represents a Kaggle dataset, with the option to define a subdirectory
    from which the data is to be extracted.
    """

    owner: str
    name: str
    subdir: Optional[str] = None

    def to_slug(self) -> str:
        """
        Returns the slugified version of the dataset name.
        """
        return f"{self.owner}-{self.name}"
