# Python Built-in Modules
import pathlib
import wave
from typing import List, Union

# Third-Party Libraries
import pandas as pd
from PIL.Image import Image
from torch.utils.data import Dataset

# Local Folders
from .meld_preprocessor_audio import MeldAudioPreprocessor
from .meld_preprocessor_text import MeldTextPreprocessor
from .meld_preprocessor_visual import MeldVisualPreprocessor
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
        self.preprocessor_text = MeldTextPreprocessor(df=self.df)
        self.preprocessor_visual = MeldVisualPreprocessor()
        self.preprocessor_audio = MeldAudioPreprocessor()

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset, based on the length
        of the dataframe itself.

        :return: The number of samples in the dataset
        """
        return len(self.df)

    def __getitem__(self, index) -> Union[MeldRecord, List[MeldRecord]]:
        """
        Returns a single sample from the dataset, based on the index provided.
        The sample is returned in the form of a `MeldRecord` object, that
        already carries the visual features (PIL.Image) and audio features
        (wave.Wave)

        :param index: The index of the sample to be returned, integer or slice
        :return: A `MeldRecord` instance or batch, containing the sample(s)
        """
        if isinstance(index, slice):
            batch = [self._get_single_item_at(i) for i in range(index.start, index.stop, index.step or 1)]
            batch = [item for item in batch if item]
            return batch
        else:
            return self._get_single_item_at(index)

    def _get_single_item_at(self, i: int):
        """
        Returns a single MELD Entry from position `i` in the dataset.

        :param i: The index of the sample to be returned
        :return: A `MeldRecord` object containing the sample
        """
        if i < len(self.df):
            row = self.df.iloc[i]

            with (
                Image.open(str(self.filedir / row.x_visual)) as image_file,
                wave.open(str(self.filedir / row.x_audio)) as audio_file,
            ):
                return MeldRecord(
                    speaker=row.speaker,
                    text=self.preprocessor_text(row),
                    visual=self.preprocessor_visual(image_file),
                    audio=self.preprocessor_audio(audio_file),
                    label=row.label,
                )
        else:
            return None

    @property
    def classes(self) -> List[str]:
        """
        Returns a list of the classes in the dataset, based on the unique
        labels in the dataframe.

        :return: A list of the classes in the dataset
        """
        return sorted(self.df.label.unique().tolist())
