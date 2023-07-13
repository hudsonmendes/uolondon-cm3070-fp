# Python Built-in Modules
import pathlib
import wave

# Third-Party Libraries
import pandas as pd
from moviepy.editor import VideoFileClip


class VideoToAudioTrackTransformer:
    """
    Creates an audio file that corresponds to the audio track of the original video.
    """

    dest: pathlib.Path
    force: bool

    def __init__(self, dest: pathlib.Path, force: bool) -> None:
        """
        Create a new audio track producer that produces an audio file from a video.
        :param dest: The destination directory to save the audio file to.
        :param n: The number of screenshots to extract from the video.
        """
        self.dest = dest
        self.force = force

    def __call__(self, row: pd.Series) -> str:
        """
        Extracts the audio track from the original .mp4 video and saves it
        to the destination directory with the specified filename.

        :param row: The row containing the filepath to the video to extract the audio track from.
        :return: The filepath of the extracted audio track.
        """

        # define the filename of the audio track
        filename, filepath = self._prepare_filepath_destination(row)

        # only writes if the force or the file does not exist
        if self.force or not filepath.exists():
            # reading videos can run into many errors, so we try to read the video
            # but get ready for not being able to read it and then we just return
            # an empty wave file.
            # we keep track of whether we managed to produce a file or not
            extracted = False
            try:
                # open the video file to extract the audio track
                # extract the audiotrack and save the audio track
                clip = VideoFileClip(str(row["x_av"]))
                audio = clip.audio
                if audio:
                    audio.write_audiofile(filepath, verbose=False, logger=None)
                    extracted = True
            except Exception as e:
                print(f"Error while reading video: {e}")

            # regardless of whether we received an error or simply could not
            # extract the audio track, we produce an empty wave file
            if not extracted:
                self._produce_empty_wave(filepath)

        return filename

    def _prepare_filepath_destination(self, row):
        filename = f"d-{row.dialogue}-seq-{row.sequence}.wav"
        filepath = self.dest / filename
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)
        return filename, filepath

    def _produce_empty_wave(self, filepath: pathlib.Path) -> None:
        """
        Produce an empty wave file at the given filepath.
        :param filepath: The filepath to produce the empty wave file at.
        :return: None.
        """
        with wave.open(str(filepath), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(44100)
            f.setnframes(0)
