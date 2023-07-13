import unittest
from unittest.mock import patch

import pathlib

from hlm12erc.etl.domain.kaggle_dataset_downloader import KaggleDatasetDownloader, KaggleDataset


class TestKaggleDatasetDownloader(unittest.TestCase):
    def setUp(self):
        self.dataset = KaggleDataset("hlm12erc", "kaggle-dataset-downloader")
        self.downloader = KaggleDatasetDownloader(self.dataset)

    @patch("hlm12erc.etl.domain.kaggle_dataset_downloader.kaggle.api.dataset_download_files")
    def test_calls_kaggle_api_with_correct_arguments(self, mock_download_files):
        # call the download method with a destination path
        dest = pathlib.Path("test_output")
        self.downloader.download(dest)

        # check that the kaggle.api.dataset_download_files function was called with the correct arguments
        mock_download_files.assert_called_once_with(
            f"{self.dataset.owner}/{self.dataset.name}",
            path=dest,
            unzip=False,
        )

    @patch("hlm12erc.etl.domain.kaggle_dataset_downloader.kaggle.api.dataset_download_files")
    def test_raises_exception_if_download_fails(self, mock_download_files):
        # configure the mock to raise an exception
        mock_download_files.side_effect = Exception("Download failed")

        # call the download method with a destination path
        dest = pathlib.Path("test_output")
        with self.assertRaises(Exception):
            self.downloader.download(dest)

    def tearDown(self):
        # remove any files created during the tests
        for filepath in pathlib.Path("test_output").glob("*"):
            filepath.unlink()
