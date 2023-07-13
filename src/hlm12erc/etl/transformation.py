# Python Built-in Modules
import logging
import pathlib
from typing import Dict, Optional

# Third-Party Libraries
import pandas as pd

# Local Folders
from .domain.filepath_recursive_discoverer import FilepathRecursiveDiscoverer
from .domain.video_to_audio_track_transformer import VideoToAudioTrackTransformer
from .domain.video_to_image_mosaic_transformer import VideoToImageMosaicTransformer

logger = logging.getLogger(__name__)


class RawTo1NFTransformer:
    """
    Transform a raw dataset into a 1NF dataset that can be
    more easily stored and consumed during training.

    Example:
        >>> from hlm12erc.etl import RawTo1NFTransformer
        >>> transformer = RawTo1NFTransformer(src="path/to/raw/dataset", workspace="path/to/workspace")
        >>> transformer.transform(dest="path/to/1nf/dataset")
    """

    DEFAULT_SNAPSHOT_HEIGHT: int = 240
    DEFAULT_SNAPSHOT_COUNT: int = 5

    src: pathlib.Path
    workspace: pathlib.Path
    n_snapshots: int
    snapshot_h: int

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
        logger.info(f"Transformation will source from: {src}")
        self.workspace = workspace / "transformer"
        logger.info(f"Transformation will use workspace: {self.workspace}")
        self.n_snapshots = n_snapshots or RawTo1NFTransformer.DEFAULT_SNAPSHOT_COUNT
        logger.info(f"Video Snapshot Count set to {self.n_snapshots}")
        self.snapshot_h = n_snapshots or RawTo1NFTransformer.DEFAULT_SNAPSHOT_HEIGHT
        logger.info(f"Video Snapshot Height set to {self.snapshot_h}")

    def transform(self, dest: pathlib.Path, force: bool) -> None:
        """
        Transform the raw dataset from the source file and save it
        to the destination file.

        :param dest: The destination file to save the transformed dataset to.
        """
        logger.info(f"Transformation will save to: {dest}")
        splis = self._get_splits()
        logger.info(f"Transformation will transform splits: {splis}")
        discover_recursively = FilepathRecursiveDiscoverer(self.src)
        for split, filename in splis.items():
            logger.info(f"Transformation will transform split: {split}")
            filepath = discover_recursively(filename)
            self._transform_split(src=filepath, dest=dest, split=split, force=force)
            logger.info(f"Split '{split}' transformed successfully")

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

    def _transform_split(self, src: pathlib.Path, dest: pathlib.Path, split: str, force: bool) -> None:
        """
        Transform the given split of the dataset.

        :param src: The source file to transform.
        :param dest: The destination file to save the transformed dataset to.
        :param split: The split to transform.
        """
        df_raw = self._collect_raw(src=src)
        df_transformed = self._collect_transformed(df=df_raw, dest=dest, force=force)
        df_ready = self._collect_wrapped(df=df_transformed)
        df_ready.to_csv(dest / f"{split}.csv", index=True)

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
                "Dialogue_ID": "dialogue",
                "Utterance_ID": "sequence",
                "Utterance": "x_text",
                "Emotion": "label",
            }
        )
        return df

    def _collect_transformed(self, df: pd.DataFrame, dest: pathlib.Path, force: bool) -> pd.DataFrame:
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
        x_av_mp4_discoverer = FilepathRecursiveDiscoverer(self.src)
        df["x_av"] = df.progress_apply(lambda row: f"dia{row.dialogue}_utt{row.sequence}.mp4", axis=1)
        df["x_av"] = df["x_av"].progress_map(x_av_mp4_discoverer)

        # produce the mosaic of images from the video, storing the mosaic into the destination folder
        df["x_visual"] = df.progress_apply(
            VideoToImageMosaicTransformer(
                dest=dest,
                n=self.n_snapshots,
                height=self.snapshot_h,
                force=force,
            ),
            axis=1,
        )

        # produce teh audio track of the video, storing audio into the destination folder
        df["x_audio"] = df.progress_apply(
            VideoToAudioTrackTransformer(
                dest=dest,
                force=force,
            ),
            axis=1,
        )
        return df

    def _collect_wrapped(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.astype(
            {
                "dialogue": "int64",
                "sequence": "int64",
                "x_text": "string",
                "x_visual": "string",
                "x_audio": "string",
                "label": "category",
            }
        )
        df = df[["dialogue", "sequence", "x_text", "x_visual", "x_audio", "label"]]
        df = df.sort_values(by=["dialogue", "sequence"])
        return df
