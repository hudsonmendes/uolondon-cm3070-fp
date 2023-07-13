import unittest

import pathlib

import pandas as pd
from moviepy.editor import VideoFileClip
from PIL import Image

from hlm12erc.etl.transformation import VideoToImageMosaicProducer


class TestVideoToImageMosaicProducer(unittest.TestCase):
    def setUp(self):
        self.outdir = pathlib.Path("/tmp/hlm12erc/test_output")
        self.producer = VideoToImageMosaicProducer(self.outdir)
        self.row = pd.Series(dict(dialog=1, seq=0, x_av="tests/fixtures/dia1_utt0.mp4"))

    def tearDown(self):
        # remove any files created during the tests
        for filepath in self.outdir.glob("*"):
            filepath.unlink()

    def test_produces_mosaic_image(self):
        # call the producer to extract the screenshots and create the mosaic image
        filename = self.producer(self.row)

        # check that the produced file exists
        filepath = self.outdir / filename
        self.assertTrue(filepath.exists())

    def test_produces_mosaic_image_with_correct_filename(self):
        # call the producer to extract the screenshots and create the mosaic image
        filename = self.producer(self.row)

        # check that the produced file has the correct filename
        expected_filename = f"d-{self.row['dialogue']}-seq-{self.row['seq']}.png"
        self.assertEqual(filename, expected_filename)

    def test_produces_mosaic_image_with_correct_number_of_screenshots(self):
        # call the producer to extract the screenshots and create the mosaic image
        self.producer.n = 5
        filename = self.producer(self.row)

        # check that the produced file has the correct number of screenshots
        filepath = self.outdir / filename
        mosaic = Image.open(filepath)
        self.assertEqual(mosaic.size[1], 5 * 1080)

    def test_produces_mosaic_image_with_correct_timestamps(self):
        # call the producer to extract the screenshots and create the mosaic image
        self.producer.n = 5
        filename = self.producer(self.row)

        # check that the produced file has the correct timestamps
        with VideoFileClip(self.row["x_av"]) as clip:
            filepath = self.outdir / filename
            mosaic = Image.open(filepath)
            timestamps = [clip.duration * i / (self.producer.n - 1) for i in range(self.producer.n)]
            for i, timestamp in enumerate(timestamps):
                screenshot = clip.get_frame(timestamp)
                expected_screenshot = Image.fromarray(screenshot)
                self.assertEqual(mosaic.crop((0, i * 1080, 1920, (i + 1) * 1080)), expected_screenshot)
