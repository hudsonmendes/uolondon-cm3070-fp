from typing import Dict, Optional

import pathlib

import pandas as pd

from .domain.recursive_filepath_discoverer import RecursiveFilepathDiscoverer
from .domain.video_filename_deducer import VideoFileNameDeducer
from .domain.video_to_audio_track_transformer import VideoToAudioTrackTransformer
from .domain.video_to_image_mosaic_transformer import VideoToImageMosaicTransformer


class RawTo1NFTransformer:
    """
    Transform a raw dataset into a 1NF dataset that can be
    more easily stored and consumed during training.
    """

    src: pathlib.Path
    workspace: pathlib.Path = pathlib.Path("/tmp/hlm12erc/etl/transformation")

    def __init__(self, src: pathlib.Path, workspace: Optional[pathlib.Path]) -> None:
        """
        Create a new transformer that transforms the raw dataset
        from the source file.
        :param src: The source file to transform.
        """
        self.src = src
        if workspace:
            self.workspace = workspace

    def transform(self, n: Optional[int]):
        """
        Transform the raw dataset from the source file and save it
        to the destination file.
        :param dest: The destination file to save the transformed dataset to.
        """
        dest = self.workspace / "transformed"
        splis = self._get_splits()
        discover_recursively = RecursiveFilepathDiscoverer(self.src)
        for split, filename in splis.items():
            filepath = discover_recursively(filename)
            self._transform_split(filepath, dest, split, n)
        return dest

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

    def _transform_split(self, src: pathlib.Path, dest: pathlib.Path, split: str, n: int) -> None:
        """
        Transform the given split of the dataset.
        :param src: The source file to transform.
        :param dest: The destination file to save the transformed dataset to.
        :param split: The split to transform.
        """
        df_raw = self._collect_raw(src=src)
        df_transformed = self._collect_transformed(df=df_raw, dest=dest, n=n)
        df_transformed.to_csv(dest / f"{split}.csv", index=False)

    def _collect_raw(self, src: pathlib.Path) -> pd.DataFrame:
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

    def _collect_transformed(self, df: pd.DataFrame, dest: pathlib.Path, n: int) -> pd.DataFrame:
        """
        Extracts image mosaic from video to produce x_visual
        and the audio track to produce x_audio and returns
        a DataFrame with the transformed data.
        """
        # deduce and locate the audio-video files from which
        # visual and audio features will be extracted
        x_av_filename_deducer = VideoFileNameDeducer()
        x_av_mp4_disoverer = RecursiveFilepathDiscoverer(self.src)
        df["x_av"] = df.apply(x_av_filename_deducer, axis=1)
        df["x_av"] = df["x_av"].map(x_av_mp4_disoverer)

        # produce the mosaic of images from the video, storing the mosaic into the destination folder
        x_visual_mosaic_producer = VideoToImageMosaicTransformer(dest=dest, n=n)
        df["x_visual"] = df.apply(x_visual_mosaic_producer, axis=1)

        # produce teh audio track of the video, storing audio into the destination folder
        x_audio_track_producer = VideoToAudioTrackTransformer(dest=dest, n=n)
        df["x_audio"] = df.apply(x_audio_track_producer, axis=1)
        return df
