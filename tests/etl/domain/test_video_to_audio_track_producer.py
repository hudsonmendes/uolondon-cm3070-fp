import unittest
import wave
from pathlib import Path

import pandas as pd

from hlm12erc.etl.transformation import VideoToAudioTrackTransformer


class TestVideoToAudioTrackProducer(unittest.TestCase):
    def setUp(self):
        self.producer = VideoToAudioTrackTransformer(Path("test_output"))

    def test_produces_audio_file(self):
        # create a test row with a video filepath
        row = pd.Series({"video_filepath": "test_video.mp4", "dialogue_id": "1", "utterance_id": "1"})

        # call the producer to extract the audio track
        filename = self.producer(row)

        # check that the produced file exists
        filepath = Path("test_output") / filename
        self.assertTrue(filepath.exists())

    def test_produces_empty_wave_file_if_no_audio(self):
        # create a test row with a video filepath that has no audio
        row = pd.Series({"video_filepath": "test_video_no_audio.mp4", "dialogue_id": "1", "utterance_id": "1"})

        # call the producer to extract the audio track
        filename = self.producer(row)

        # check that the produced file exists and is an empty wave file
        filepath = Path("test_output") / filename
        self.assertTrue(filepath.exists())
        with wave.open(str(filepath), "rb") as f:
            self.assertEqual(f.getnframes(), 0)

    def test_produces_empty_wave_file_if_audio_is_none(self):
        # create a test row with a video filepath and None for the audio track
        row = pd.Series({"video_filepath": "test_video.mp4", "dialogue_id": "1", "utterance_id": "1"})
        row["audio"] = None

        # call the producer to extract the audio track
        filename = self.producer(row)

        # check that the produced file exists and is an empty wave file
        filepath = Path("test_output") / filename
        self.assertTrue(filepath.exists())
        with wave.open(str(filepath), "rb") as f:
            self.assertEqual(f.getnframes(), 0)

    def test_produces_audio_file_with_correct_filename(self):
        # create a test row with a video filepath
        row = pd.Series({"video_filepath": "test_video.mp4", "dialogue_id": "1", "utterance_id": "1"})

        # call the producer to extract the audio track
        filename = self.producer(row)

        # check that the produced file has the correct filename
        expected_filename = f"d-{row['dialogue_id']}-seq-{row['utterance_id']}.wav"
        self.assertEqual(filename, expected_filename)

    def tearDown(self):
        # remove any files created during the tests
        for filepath in Path("test_output").glob("*"):
            filepath.unlink()
