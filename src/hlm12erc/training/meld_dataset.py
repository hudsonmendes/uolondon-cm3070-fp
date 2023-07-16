# Python Built-in Modules
import pathlib
import wave
from collections import namedtuple
from typing import NamedTuple

# Third-Party Libraries
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Local Folders
from .meld_record import MeldDialogueEntry, MeldRecord


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
        sequence = row["sequence"]
        previous_dialogue = self._extract_previous_dialogue(
            dialogue=dialogue,
            before=sequence,
        )
        return MeldRecord(
            speaker=row.speaker,
            visual=Image.open(str(self.filedir / row.x_visual)),
            audio=wave.open(str(self.filedir / row.x_audio)),
            previous_dialogue=previous_dialogue,
            utterance=row.x_text,
            label=row.label,
        )

    def _extract_previous_dialogue(self, dialogue: int, before: int) -> List[MeldDialogueEntry]:
        """
        Extracts the dialogue that happened before the current one, based on
        the dialogue number and the sequence number of the current dialogue.

        :param dialogue: The dialogue number of the current dialogue
        :param before: The sequence number of the current dialogue
        :return: The previous dialogue as a list of `MeldDialogueEntry`
        """
        previous_dialogue = self.df[(self.df.dialogue == dialogue) & (self.df.sequence < before)]
        return [MeldDialogueEntry(speaker=row.speaker, utterance=row.x_text) for _, row in previous_dialogue.iterrows()]
