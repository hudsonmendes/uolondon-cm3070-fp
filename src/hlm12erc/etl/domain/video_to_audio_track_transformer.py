import pathlib
import wave
from typing import Optional

import pandas as pd
from moviepy.editor import VideoFileClip


class VideoToAudioTrackTransformer:
    """
    Creates an audio file that corresponds to the audio track of the original video.
    """

    dest: pathlib.Path

    def __init__(self, dest: pathlib.Path) -> None:
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
        clip = VideoFileClip(row["x_av"])

        # extract the audiotrack
        audio = clip.audio

        # save the audio track or an empty wave to the destination directory with the specified filename
        filename = f"d-{row['dialogue']}-seq-{row['seq']}.wav"
        filepath = self.dest / filename
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)
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
