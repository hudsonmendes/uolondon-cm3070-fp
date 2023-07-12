import pathlib
import shutil
import unittest

from hlm12erc.etl import KaggleDownloader


class TestKaggleDownloader(unittest.TestCase):
    def setUp(self):
        self.test_dir = pathlib.Path("test_data")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_download_to_file(self):
        # download the dataset to a file instead of a directory
        # and check that the file exists
        downloader = KaggleDownloader("crawford", "80-cereals")
        downloader.download(self.test_dir / "crawford-80cereals.zip")
        self.assertTrue((self.test_dir / "crawford-80cereals.zip").exists())

    def test_download_invalid(self):
        # download a dataset that does not exist to check failure
        downloader = KaggleDownloader("does-not", "exist")
        with self.assertRaises(Exception):
            downloader.download(self.test_dir / "does-not-exist.zip")
