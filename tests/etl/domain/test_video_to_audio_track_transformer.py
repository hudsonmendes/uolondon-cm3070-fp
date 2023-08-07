# Python Built-in Modules
import pathlib
import shutil
import unittest
import wave

# Third-Party Libraries
import pandas as pd


class TestVideoToAudioTrackTransformer(unittest.TestCase):
    def setUp(self):
        self.mp4 = pathlib.Path("tests/fixtures/dia1_utt0.mp4")
        self.wav = pathlib.Path("tests/fixtures/dia1_utt0.wav")
        self.dest = pathlib.Path("/tmp/hlm12erc/test_output/")
        self.dest.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.dest)

    def test_audio_extraction(self):
        # create a test row with the filepath to the test video clip
        row = pd.Series({"x_av": str(self.mp4), "dialogue": 1, "sequence": 1})

        # create a transformer instance and call it on the test row
        transformer = self._create_subject(force=True)
        filename = transformer(row)

        # check that the audio track was extracted and saved to the destination directory
        filepath = self.dest / filename
        self.assertTrue(filepath.exists())
        with wave.open(str(filepath), "rb") as f1, wave.open(str(self.wav), "rb") as f2:
            self.assertEqual(f1.getnchannels(), f2.getnchannels())
            self.assertEqual(f1.getsampwidth(), f2.getsampwidth())
            self.assertEqual(f1.getframerate(), f2.getframerate())
            self.assertEqual(f1.getnframes(), f2.getnframes())
            self.assertEqual(f1.readframes(-1), f2.readframes(-1))

    def test_empty_wave_production(self):
        # create a test row with an invalid filepath
        noise = pathlib.Path("tests/fixtures/noise_no_audio.mp4")
        row = pd.Series({"x_av": str(noise), "dialogue": 1, "sequence": 1})

        # create a transformer instance and call it on the test row
        transformer = self._create_subject(force=True)
        filename = transformer(row)

        # check that an empty wave file was produced and saved to the destination directory
        filepath = self.dest / filename
        self.assertTrue(filepath.exists())
        with wave.open(str(filepath), "rb") as f:
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getsampwidth(), 2)
            self.assertEqual(f.getframerate(), 44100)
            self.assertEqual(f.getnframes(), 0)

    def _create_subject(self, force):
        # local import to avoid etl dependencies becoming global requirements
        # My Packages and Modules
        from hlm12erc.etl.domain.video_to_audio_track_transformer import (
            VideoToAudioTrackTransformer,
        )

        # create and return a transformer instance
        return VideoToAudioTrackTransformer(self.dest, force=force)
