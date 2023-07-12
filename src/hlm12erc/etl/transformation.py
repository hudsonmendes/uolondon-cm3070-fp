import pathlib

from typing import Union, Optional, Dict

import pandas as pd

from .utils import ensure_path


class RawTo1NFTransformer:
    """
    Transform a raw dataset into a 1NF dataset that can be
    more easily stored and consumed during training.
    """

    src: pathlib.Path
    n_screenshots: Optional[int] = None

    def __init__(self, src: Union[str, pathlib.Path]) -> None:
        """
        Create a new transformer that transforms the raw dataset
        from the source file.
        :param src: The source file to transform.
        """
        self.src = pathlib.Path(src)

    def extracting_n_screenshots(self, n: Optional[int]) -> "RawTo1NFTransformer":
        """
        Set the number of screenshots to extract from each video.
        :param n: The number of screenshots to extract.
        :return: The transformer instance.
        """
        self.n_screenshots = n if (n and n > 0) else None
        return self

    def transform(self, dest: Union[str, pathlib.Path]) -> None:
        """
        Transform the raw dataset from the source file and save it
        to the destination file.
        :param dest: The destination file to save the transformed dataset to.
        """
        dest = ensure_path(dest)
        splis = self._get_splits()
        for split, filename in splis.items():
            filepath = self._discover_filepath_for(filename)
            self._transform_split(filepath, dest, split)

    def _get_splits(self) -> Dict[str, str]:
        """
        Get the splits of the dataset, and their corresponding filenames.
        :return: A dictionary containing the splits of the dataset.
        """
        return {
            "train": "train_sent_emo.csv",
            "valid": "dev_sent_emo.csv",
            "test": "test_sent_emo.csv",
        }

    def _transform_split(self, src: pathlib.Path, dest: pathlib.Path, split: str) -> None:
        """
        Transform the given split of the dataset.
        :param src: The source file to transform.
        :param dest: The destination file to save the transformed dataset to.
        :param split: The split to transform.
        """
        df_raw = self._load_from_raw(src)

    def _load_from_raw(self, src: pathlib.Path):
        df = pd.read_csv(src)
        df["VideoFilename"] = df.apply(self._videofilename_from, axis=1)
        df = df.rename(columns={
            "Dialogue_ID": "dialogue",
            "Utterance_ID": "sequence",
            "Utterance": "x_text",
            "VideoFilename": "x_video",
            "Emotion": "label",
        }
        return df

    def _discover_filepath_for(self, filename: str) -> pathlib.Path:
        """
        Searches the `src` folder recursively to find a filename
        that matches the given filename, and returns the pathlib.Path
        instance of that file.
        :param filename: The filename to search for.
        :return: The pathlib.Path instance of the file.
        """
        for path in self.src.rglob(filename):
            return path
        raise FileNotFoundError(f"Could not find {filename} in {self.src}")

    def _videofilename_from(self, row: pd.Series) -> str:
        """
        Get the video filename from the given row.
        :param row: The row to get the video filename from.
        :return: The video filename.
        """
        filename = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}.mp4"
        return str(self._discover_filepath_for(filename))
