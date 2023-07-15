# Python Built-in Modules
import wave

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

    def __init__(self, df: pd.DataFrame):
        """
        Creates a new instance of the MeldDataset for a split

        :param df: The dataframe containing the data for the split
        """
        df = df.sort_values(by=["dialogue", "sequence"], ascending=[True, True])
        self.df = df

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
        previous_utterances = df_prev.utterance.tolist()

        return MeldRecord(
            visual=Image.open(row.x_visual),
            audio=wave.open(row.x_audio),
            dialogue=previous_utterances,
            utterance=row.utterance,
        )