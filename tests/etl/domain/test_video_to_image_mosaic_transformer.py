# Python Built-in Modules
import pathlib
import shutil
import unittest

# Third-Party Libraries
import pandas as pd
from PIL import Image


class TestVideoToImageMosaicTransformer(unittest.TestCase):
    def setUp(self):
        self.dest = pathlib.Path("/tmp/hlm12erc/tests/test_output")
        self.dest.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.dest)

    def test_screenshot_extraction(self):
        # create a test row with the filepath to the test video clip
        row = pd.Series({"x_av": "tests/fixtures/dia1_utt0.mp4", "dialogue": 1, "sequence": 1})

        # create a transformer instance and call it on the test row
        transformer = self._create_subject(n=3, height=100, force=True)
        filename = transformer(row)

        # check that the mosaic image was extracted and saved to the destination directory
        filepath = self.dest / filename
        self.assertTrue(filepath.exists())

    def test_screenshot_extraction_number(self):
        # create a test row with the filepath to the test video clip
        row = pd.Series({"x_av": "tests/fixtures/dia1_utt0.mp4", "dialogue": 1, "sequence": 1})

        # create a transformer instance and call it on the test row
        transformer = self._create_subject(n=5, height=100, force=True)
        filename = transformer(row)

        # check that the mosaic image was extracted and saved to the destination directory
        filepath = self.dest / filename
        self.assertTrue(filepath.exists())

    def test_screenshot_extraction_filename(self):
        # create a test row with the filepath to the test video clip
        row = pd.Series({"x_av": "tests/fixtures/dia1_utt0.mp4", "dialogue": 1, "sequence": 1})

        # create a transformer instance and call it on the test row
        transformer = self._create_subject(n=3, height=100, force=True)
        filename = transformer(row)

        # check that the mosaic image was extracted and saved to the destination directory with the correct filename
        filepath = self.dest / filename
        self.assertEqual(filepath.name, "d-1-seq-1.png")

    def test_screenshot_extraction_dimensions(self):
        # create a test row with the filepath to the test video clip
        row = pd.Series({"x_av": "tests/fixtures/dia1_utt0.mp4", "dialogue": 1, "sequence": 1})

        # define the number of snapshots to extract from the video
        n_snapshots = 5
        snapshot_height = 100

        # create a transformer instance and call it on the test row
        transformer = self._create_subject(n=n_snapshots, height=snapshot_height, force=True)
        filename = transformer(row)

        # check that the mosaic image has the correct dimensions
        filepath = self.dest / filename
        mosaic = Image.open(filepath)
        self.assertEqual(mosaic.size[1], snapshot_height * n_snapshots)

    def _create_subject(self, n: int, height: int, force: bool):
        # local import to avoid etl dependencies becoming global requirements
        # My Packages and Modules
        from hlm12erc.etl.domain.video_to_image_mosaic_transformer import (
            VideoToImageMosaicTransformer,
        )

        # create and return instance
        return VideoToImageMosaicTransformer(self.dest, n=n, height=height, force=force)
