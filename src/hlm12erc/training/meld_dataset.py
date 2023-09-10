# Python Built-in Modules
import pathlib
from typing import List

# Third-Party Libraries
import pandas as pd
import torch
from torch.utils.data import Dataset

# Local Folders
from .meld_record import MeldRecord
from .meld_record_preprocessor_audio import MeldAudioPreprocessor, MeldAudioPreprocessorToWaveform
from .meld_record_preprocessor_text import MeldTextPreprocessor, MeldTextPreprocessorToDialogPrompt
from .meld_record_preprocessor_visual import MeldVisualPreprocessor, MeldVisualPreprocessorFilepathToResnet50


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
    df: pd.DataFrame
    classes_: List[str]
    preprocessors_text: List[MeldTextPreprocessor]
    preprocessors_visual: List[MeldVisualPreprocessor]
    preprocessors_audio: List[MeldAudioPreprocessor]
    inhibit_text: bool = False
    inhibit_visual: bool = False
    inhibit_audio: bool = False

    def __init__(
        self,
        filepath: pathlib.Path,
        filedir: pathlib.Path | None = None,
        df: pd.DataFrame | None = None,
        classes: List[str] | None = None,
        preprocessors_text: List[MeldTextPreprocessor] | None = None,
        preprocessors_visual: List[MeldVisualPreprocessor] | None = None,
        preprocessors_audio: List[MeldAudioPreprocessor] | None = None,
        inhibit_text: bool = False,
        inhibit_visual: bool = False,
        inhibit_audio: bool = False,
    ):
        """
        Creates a new instance of the MeldDataset for a split

        :param filepath: The dataframe containing the data for the split
        :param filedir: The directory where the files are located, None leads to default
        :param df: The dataframe containing the data for the split
        :param classes: the list of classes in the dataset, None leads to default
        :param preprocessors_text: the list of text preprocessors to be applied, None leads to default
        :param preprocessors_visual: the list of visual preprocessors to be applied, None leads to default
        :param preprocessors_audio: the list of audio preprocessors to be applied, None leads to default
        :param inhibit_text: Whether to inhibit the text preprocessing
        :param inhibit_visual: Whether to inhibit the visual preprocessing
        :param inhibit_audio: Whether to inhibit the audio preprocessing
        """
        self.filepath = filepath
        self.filedir = filedir or filepath.parent
        self.df = df
        self.classes_ = []
        if classes is not None:
            self.classes_ = classes
        if self.df is None:
            self.df = pd.read_csv(self.filepath).sort_values(by=["dialogue", "sequence"], ascending=[True, True])
        if self.classes_ is None:
            self.classes_ = sorted(self.df.label.unique())
        self.preprocessors_text = preprocessors_text or [MeldTextPreprocessorToDialogPrompt(df=self.df)]
        self.preprocessors_visual = preprocessors_visual or [MeldVisualPreprocessorFilepathToResnet50()]
        self.preprocessors_audio = preprocessors_audio or [MeldAudioPreprocessorToWaveform()]
        self.inhibit_text = inhibit_text
        self.inhibit_visual = inhibit_visual
        self.inhibit_audio = inhibit_audio

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset, based on the length
        of the dataframe itself.

        :return: The number of samples in the dataset
        """
        return len(self.df)

    def __getitem__(
        self,
        index: slice | int,
    ) -> MeldRecord | List[MeldRecord]:
        # for batch/slices, we recursively return this call to getitem with the integer index
        if isinstance(index, slice):
            records = [self._read(i, out_of_range_ok=True) for i in range(index.start, index.stop, index.step or 1)]
            return [r for r in records if isinstance(r, MeldRecord) and r is not None]
        # for single items, we return the record at that index and raise an error if out of range
        else:
            return self._read(index=index, out_of_range_ok=False)

    def _read(
        self,
        index: int,
        out_of_range_ok: bool,
    ) -> MeldRecord:
        """
        Returns a single sample from the dataset, based on the index provided.
        The sample is returned in the form of a `MeldRecord` object, that
        already carries the visual features (PIL.Image) and audio features
        (wave.Wave)

        :param index: The index of the sample to be returned, integer or slice
        :param out_of_range_ok: Whether to return None or raise an error when the index is out of range, used for slices
        :return: A `MeldRecord` instance or batch, containing the sample(s)
        """
        # when the index is precise, we return the record at that index
        if index < len(self.df):
            row = self.df.iloc[index]
            text = None
            if not self.inhibit_text:
                text = self._preprocess_text(row)
            visual = None
            if not self.inhibit_visual:
                visual = self._preprocess_visual(row)
            audio = None
            if not self.inhibit_audio:
                self._preprocess_audio(row)
            return MeldRecord(text=text, visual=visual, audio=audio, label=row.label)
        elif not out_of_range_ok:
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.df)}")

    def clone_inhibiting(self, text: bool = False, visual: bool = False, audio: bool = False) -> "MeldDataset":
        """
        Clones the preset dataset, inhibiting the preprocessing of one or more modalities,
        optimising for processing with less modalities than the 3 available.

        :param text: Whether to inhibit the text preprocessing
        :param visual: Whether to inhibit the visual preprocessing
        :param audio: Whether to inhibit the audio preprocessing
        """
        return MeldDataset(
            filepath=self.filepath,
            filedir=self.filedir,
            df=self.df,
            classes=self.classes_,
            preprocessors_text=self.preprocessors_text,
            preprocessors_visual=self.preprocessors_visual,
            preprocessors_audio=self.preprocessors_audio,
            inhibit_text=text,
            inhibit_visual=visual,
            inhibit_audio=audio,
        )

    def preprocessing_with(self, *preprocessors: list) -> "MeldDataset":
        """
        Returns a copy of the present dataset, with the preprocessors preprended
        to the list of preprocessors already present in the dataset.

        :param preprocessors: The list of preprocessors to be applied
        :return: A new instance (copy) of the dataset, with the preprocessors prepended
        """
        new_pp_text = [fn for fn in preprocessors if isinstance(fn, MeldTextPreprocessor)]
        new_pp_visual = [fn for fn in preprocessors if isinstance(fn, MeldVisualPreprocessor)]
        new_pp_audio = [fn for fn in preprocessors if isinstance(fn, MeldAudioPreprocessor)]
        return MeldDataset(
            filepath=self.filepath,
            filedir=self.filedir,
            df=self.df,
            classes=self.classes_,
            preprocessors_text=new_pp_text + self.preprocessors_text,
            preprocessors_visual=new_pp_visual + self.preprocessors_visual,
            preprocessors_audio=new_pp_audio + self.preprocessors_audio,
        )

    def _preprocess_text(self, row: pd.Series) -> str:
        """
        Applies each one of the text preprocessors and returns the last output

        :param row: The row to be preprocessed
        :return: The preprocessed string
        """
        y = row
        for fn in self.preprocessors_text:
            y = fn(y)
        return y

    def _preprocess_visual(self, row: pd.Series) -> torch.Tensor:
        """
        Applies each one of the visual preprocessors and returns the last output

        :param row: The row to be preprocessed
        :return: The preprocessed tensor
        """
        y = self.filedir / row.x_visual
        for fn in self.preprocessors_visual:
            y = fn(y)
        return y

    def _preprocess_audio(self, row: pd.Series) -> torch.Tensor:
        """
        Applies each one of the audio preprocessors and returns the last output

        :param row: The row to be preprocessed
        :return: The preprocessed tensor
        """
        y = self.filedir / row.x_audio
        for fn in self.preprocessors_audio:
            y = fn(y)
        return y

    @property
    def classes(self) -> List[str]:
        """
        Returns a list of the classes in the dataset, based on the unique
        labels in the dataframe.

        :return: A list of the classes in the dataset
        """
        return self.classes_

    @property
    def labels(self) -> List[str]:
        """
        Returns a list of the labels in the dataset, based on the unique
        labels in the dataframe.

        :return: A list of the labels in the dataset
        """
        return list(self.df.label)
