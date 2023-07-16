# Python Built-in Modules
import pathlib
import wave
from typing import Union

# Third-Party Libraries
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Local Folders
from .meld_record import MeldRecord


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

    def __init__(self, filepath: pathlib.Path):
        """
        Creates a new instance of the MeldDataset for a split

        :param df: The dataframe containing the data for the split
        """
        self.filepath = filepath
        self.filedir = filepath.parent
        self.df = pd.read_csv(self.filepath).sort_values(by=["dialogue", "sequence"], ascending=[True, True])

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset, based on the length
        of the dataframe itself.

        :return: The number of samples in the dataset
        """
        return len(self.df)

    def __getitem__(self, index) -> MeldRecord:
        """
        Returns a single sample from the dataset, based on the index provided.
        The sample is returned in the form of a `MeldRecord` object, that
        already carries the visual features (PIL.Image) and audio features
        (wave.Wave)

        :param index: The index of the sample to be returned
        :return: A `MeldRecord` object containing the sample
        """
        row = self.df.iloc[index]

        dialogue = row["dialogue"]
        df_prev = self.df[self.df.dialogue == dialogue]
        df_prev = df_prev[df_prev.sequence < row.sequence]
        previous_utterances = df_prev.x_text.tolist()

        return MeldRecord(
            visual=Image.open(str(self.filedir / row.x_visual)),
            audio=wave.open(str(self.filedir / row.x_audio)),
            dialogue=previous_utterances,
            utterance=row.x_text,
        )
