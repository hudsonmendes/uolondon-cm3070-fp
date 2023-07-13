import pathlib
from typing import Dict, Optional

import pandas as pd

from .domain.filepath_recursive_discoverer import FilepathRecursiveDiscoverer
from .domain.video_filename_deducer import VideoFileNameDeducer
from .domain.video_to_audio_track_transformer import VideoToAudioTrackTransformer
from .domain.video_to_image_mosaic_transformer import VideoToImageMosaicTransformer


class RawTo1NFTransformer:
    """
    Transform a raw dataset into a 1NF dataset that can be
    more easily stored and consumed during training.

    Example:
        >>> from hlm12erc.etl import RawTo1NFTransformer
        >>> transformer = RawTo1NFTransformer(src="path/to/raw/dataset", workspace="path/to/workspace")
        >>> transformer.transform(dest="path/to/1nf/dataset")
    """

    DEFAULT_N_SNAPSHOTS: int = 3

    src: pathlib.Path
    n_snapshots: int
    workspace: pathlib.Path

    def __init__(
        self,
        src: pathlib.Path,
        workspace: pathlib.Path,
        n_snapshots: Optional[int] = None,
    ) -> None:
        """
        Create a new transformer that transforms the raw dataset
        from the source file.

        :param src: The source file to transform.
        :param workspace: The workspace that will be used for processing.
        :param n_snapshots: The number of snapshots to take from each video.
        """
        self.src = src
        self.workspace = workspace / "transformer"
        self.n_snapshots = n_snapshots or RawTo1NFTransformer.DEFAULT_N_SNAPSHOTS

    def transform(self, dest: pathlib.Path) -> None:
        """
        Transform the raw dataset from the source file and save it
        to the destination file.

        :param dest: The destination file to save the transformed dataset to.
        """
        splis = self._get_splits()
        discover_recursively = FilepathRecursiveDiscoverer(self.src)
        for split, filename in splis.items():
            filepath = discover_recursively(filename)
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
        df_raw = self._collect_raw(src=src)
        df_transformed = self._collect_transformed(df=df_raw, dest=dest)
        df_transformed.to_csv(dest / f"{split}.csv", index=False)

    def _collect_raw(self, src: pathlib.Path) -> pd.DataFrame:
        """
        Renames the columns to something simpler and returns
        the dataframe with the raw data

        :param src: The source file to collect the raw data from.
        :return: A DataFrame containing the raw data.
        """
        df = pd.read_csv(src)
        df = df.rename(
            columns={
                "dialogue": "dialogue",
                "Utterance_ID": "sequence",
                "Utterance": "x_text",
                "Emotion": "label",
            }
        )
        return df

    def _collect_transformed(self, df: pd.DataFrame, dest: pathlib.Path) -> pd.DataFrame:
        """
        Extracts image mosaic from video to produce x_visual
        and the audio track to produce x_audio and returns
        a DataFrame with the transformed data.

        :param df: The DataFrame containing the raw data.
        :param dest: The destination folder where the transformed data will be stored.
        :return: A DataFrame containing the transformed data.
        """
        # deduce and locate the audio-video files from which
        # visual and audio features will be extracted
        x_av_filename_deducer = VideoFileNameDeducer()
        x_av_mp4_disoverer = FilepathRecursiveDiscoverer(self.src)
        df["x_av"] = df.apply(x_av_filename_deducer, axis=1)
        df["x_av"] = df["x_av"].map(x_av_mp4_disoverer)

        # produce the mosaic of images from the video, storing the mosaic into the destination folder
        x_visual_mosaic_producer = VideoToImageMosaicTransformer(dest=dest, n=self.n_snapshots)
        df["x_visual"] = df.apply(x_visual_mosaic_producer, axis=1)

        # produce teh audio track of the video, storing audio into the destination folder
        x_audio_track_producer = VideoToAudioTrackTransformer(dest=dest)
        df["x_audio"] = df.apply(x_audio_track_producer, axis=1)
        return df
