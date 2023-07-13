from typing import Union, Optional, Dict

import wave
import pathlib

import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image

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
        discover_recursively = RecursiveFilepathDiscoverer(self.src)
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

    def _collect_raw(self, src: pathlib.Path) -> pd.DataFrame:
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

    def _collect_transformed(self, df: pd.DataFrame, dest: pathlib.Path) -> pd.DataFrame:
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
        x_visual_mosaic_producer = VideoToImageMosaicProducer(dest=dest, n=self.n_screenshots)
        df["x_visual"] = df.apply(x_visual_mosaic_producer, axis=1)

        # produce teh audio track of the video, storing audio into the destination folder
        x_audio_track_producer = VideoToAudioTrackProducer(dest=dest, n=self.n_screenshots)
        df["x_audio"] = df.apply(x_audio_track_producer, axis=1)
        return df


class VideoFileNameDeducer:
    """
    Deduce the video filename from the given row.
    """

    def __call__(self, row: pd.Series) -> str:
        """
        Get the video filename from the given row.
        :param row: The row to get the video filename from.
        :return: The video filename.
        """
        return f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}.mp4"


class RecursiveFilepathDiscoverer:
    """
    Discover the filepath for the given filename.
    """

    root: pathlib.Path

    def __init__(self, root: pathlib.Path) -> None:
        """
        Create a new filepath discoverer looking at the given root directory.
        :param root: The root directory to discover the filepath from.
        """
        self.root = root

    def __call__(self, filename: str) -> pathlib.Path:
        """
        Discover the filepath for the given filename.
        :param filename: The filename to discover the filepath for.
        :return: The filepath for the given filename.
        """
        for path in self.root.rglob(filename):
            return path
        raise FileNotFoundError(f"Could not find {filename} in {self.root}")


class VideoToImageMosaicProducer:
    """
    Produce a mosaic of images from a video.
    """

    dest: pathlib.Path
    n: Optional[int] = None

    def __init__(self, dest: pathlib.Path, n: Optional[int] = None) -> None:
        """
        Create a new mosaic producer that produces a mosaic of images
        from a video.
        :param dest: The destination directory to save the mosaic image to.
        :param n: The number of screenshots to extract from the video.
        """
        self.dest = dest
        self.n = n

    def __call__(self, filepath: str, dialogue_id: str, utterance_id: str) -> str:
        """
        Extracts a number of screenshots defined by `self.n` from the original
        .mp4 video, equidistant to one another.

        :param filepath: The filepath to the video to extract the screenshots from.
        :param dialogue_id: The dialogue ID to use in the mosaic filename.
        :param utterance_id: The utterance ID to use in the mosaic filename.
        :return: The filepath of the extracted screenshots mosaic image.
        """
        # open the video file to extract the screenshots
        clip = VideoFileClip(filepath)

        # find the duration in seconds of the video clip
        duration = clip.duration

        # calculate the timestamps for the screenshots
        n_screenshots = self.n if (self.n and self.n > 0) else 3
        timestamps = [duration * i / (n_screenshots - 1) for i in range(n_screenshots)]

        # extract the screenshots and stack them on top of each other
        screenshots = []
        for timestamp in timestamps:
            screenshot = clip.get_frame(timestamp)
            screenshots.append(Image.fromarray(screenshot))

        mosaic = Image.new("RGB", (screenshots[0].width, screenshots[0].height * n_screenshots))
        for i, screenshot in enumerate(screenshots):
            mosaic.paste(screenshot, (0, i * screenshot.height))

        # save the mosaic image to the destination directory with the specified filename
        filename = f"d-{dialogue_id}-seq-{utterance_id}.png"
        mosaic.save(self.dest / filename)
        return filename



class VideoToAudioTrackProducer:
    """
    Creates an audio file that corresponds to the audio track of the original video.
    """

    dest: pathlib.Path

    def __init__(self, dest: pathlib.Path, n: Optional[int] = None) -> None:
        """
        Create a new audio track producer that produces an audio file from a video.
        :param dest: The destination directory to save the audio file to.
        :param n: The number of screenshots to extract from the video.
        """
        self.dest = dest

    def __call__(self, row: pd.Series) -> str:
        """
        Extracts the audio track from the original .mp4 video and saves it
        to the destination directory with the specified filename.

        :param row: The row containing the filepath to the video to extract the audio track from.
        :return: The filepath of the extracted audio track.
        """
        # open the video file to extract the audio track
        clip = VideoFileClip(row["video_filepath"])

        # extract the audiotrack
        audio = clip.audio

        # save the audio track or an empty wave to the destination directory with the specified filename
        filename = f"d-{row['dialogue_id']}-seq-{row['utterance_id']}.wav"
        filepath = self.dest / filename
        audio.write_audiofile(filepath) if audio else self._produce_empty_wave(filepath)
        return filename

    def _produce_empty_wave(self, filepath: pathlib.Path) -> None:
        """
        Produce an empty wave file at the given filepath.
        :param filepath: The filepath to produce the empty wave file at.
        :return: None.
        """
        with wave.open(str(filepath), "wb+") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(44100)
            f.setnframes(0)
