# Python Built-in Modules
import pathlib
from typing import List, Optional

# Third-Party Libraries
import pandas as pd
import torch
from torch.utils.data import Dataset

# Local Folders
from .meld_record import MeldRecord
from .meld_record_reader import MeldRecordReader


class MeldDataset(Dataset):
    """
    Using a dataframe to the dataset .csv file, this class allows access to
    the data in the high level form of a `MeldRecord` object, that already
    carries the visual features (PIL.Image) and audio features (wave.Wave).

    Example:
        >>> df = pd.read_csv("data/meld/train.csv")
        >>> dataset = MeldDataset(df)
        >>> record = dataset[0]
    """

    filepath: pathlib.Path
    filedir: pathlib.Path
    records: List[MeldRecord]

    def __init__(self, filepath: pathlib.Path, device: Optional[torch.device] = None):
        """
        Creates a new instance of the MeldDataset for a split

        :param df: The dataframe containing the data for the split
        """
        self.filepath = filepath
        self.filedir = filepath.parent
        df = pd.read_csv(self.filepath).sort_values(by=["dialogue", "sequence"], ascending=[True, True])
        record_reader = MeldRecordReader(df=df, filename=filepath.stem, filedir=self.filedir, device=device)
        self.records = record_reader.read_all_valid()

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset, based on the length
        of the dataframe itself.

        :return: The number of samples in the dataset
        """
        return len(self.records)

    def __getitem__(self, index: slice | int) -> MeldRecord | List[MeldRecord]:
        """
        Returns a single sample from the dataset, based on the index provided.
        The sample is returned in the form of a `MeldRecord` object, that
        already carries the visual features (PIL.Image) and audio features
        (wave.Wave)

        :param index: The index of the sample to be returned, integer or slice
        :return: A `MeldRecord` instance or batch, containing the sample(s)
        """
        if isinstance(index, slice):
            return [self.records[i] for i in range(index.start, index.stop, index.step or 1)]
        elif isinstance(index, int):
            return self.records[index]
        else:
            raise TypeError(f"The index '{index}' is an invalid index type.")

    @property
    def classes(self) -> List[str]:
        """
        Returns a list of the classes in the dataset, based on the unique
        labels in the dataframe.

        :return: A list of the classes in the dataset
        """
        return sorted(self.df.label.unique().tolist())
