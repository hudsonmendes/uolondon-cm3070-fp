import unittest
from pathlib import Path

import pandas as pd
from moviepy.editor import VideoFileClip
from PIL import Image

from hlm12erc.etl.transformation import VideoToImageMosaicProducer


class TestVideoToImageMosaicProducer(unittest.TestCase):
    def setUp(self):
        self.producer = VideoToImageMosaicProducer(Path("test_output"))

    def test_produces_mosaic_image(self):
        # create a test row with a video filepath
        row = pd.Series({"video_filepath": "test_video.mp4", "dialogue_id": "1", "utterance_id": "1"})

        # call the producer to extract the screenshots and create the mosaic image
        filename = self.producer(row["video_filepath"], row["dialogue_id"], row["utterance_id"])

        # check that the produced file exists
        filepath = Path("test_output") / filename
        self.assertTrue(filepath.exists())

    def test_produces_mosaic_image_with_correct_filename(self):
        # create a test row with a video filepath
        row = pd.Series({"video_filepath": "test_video.mp4", "dialogue_id": "1", "utterance_id": "1"})

        # call the producer to extract the screenshots and create the mosaic image
        filename = self.producer(row["video_filepath"], row["dialogue_id"], row["utterance_id"])

        # check that the produced file has the correct filename
        expected_filename = f"d-{row['dialogue_id']}-seq-{row['utterance_id']}.png"
        self.assertEqual(filename, expected_filename)

    def test_produces_mosaic_image_with_correct_number_of_screenshots(self):
        # create a test row with a video filepath
        row = pd.Series({"video_filepath": "test_video.mp4", "dialogue_id": "1", "utterance_id": "1"})

        # call the producer to extract the screenshots and create the mosaic image
        self.producer.n = 5
        filename = self.producer(row["video_filepath"], row["dialogue_id"], row["utterance_id"])

        # check that the produced file has the correct number of screenshots
        filepath = Path("test_output") / filename
        mosaic = Image.open(filepath)
        self.assertEqual(mosaic.size[1], 5 * 1080)

    def test_produces_mosaic_image_with_correct_timestamps(self):
        # create a test row with a video filepath
        row = pd.Series({"video_filepath": "test_video.mp4", "dialogue_id": "1", "utterance_id": "1"})

        # create a moviepy VideoFileClip object
        clip = VideoFileClip(row["video_filepath"])

        # call the producer to extract the screenshots and create the mosaic image
        self.producer.n = 5
        filename = self.producer(row["video_filepath"], row["dialogue_id"], row["utterance_id"])

        # check that the produced file has the correct timestamps
        filepath = Path("test_output") / filename
        mosaic = Image.open(filepath)
        timestamps = [clip.duration * i / (self.producer.n - 1) for i in range(self.producer.n)]
        for i, timestamp in enumerate(timestamps):
            screenshot = clip.get_frame(timestamp)
            expected_screenshot = Image.fromarray(screenshot)
            self.assertEqual(mosaic.crop((0, i * 1080, 1920, (i + 1) * 1080)), expected_screenshot)

        # close the clip
        clip.close()

    def tearDown(self):
        # remove any files created during the tests
        for filepath in Path("test_output").glob("*"):
            filepath.unlink()
