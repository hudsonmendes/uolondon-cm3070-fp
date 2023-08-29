# Python Built-in Modules
import pathlib
import unittest
from unittest.mock import patch


class TestKaggleDatasetDownloader(unittest.TestCase):
    def setUp(self):
        # local import to avoid etl dependencies becoming global requirements
        # My Packages and Modules
        from hlm12erc.etl.domain.kaggle_dataset_downloader import KaggleDataset, KaggleDatasetDownloader

        self.dataset = KaggleDataset("hlm12erc", "kaggle-dataset-downloader")
        self.downloader = KaggleDatasetDownloader(self.dataset)

    @patch("hlm12erc.etl.domain.kaggle_dataset_downloader.kaggle.api.dataset_download_files")
    def test_calls_kaggle_api_with_correct_arguments(self, mock_download_files):
        # call the download method with a destination path
        dest = pathlib.Path("test_output")
        self.downloader.download(dest, force=True)

        # check that the kaggle.api.dataset_download_files function was called with the correct arguments
        mock_download_files.assert_called_once_with(
            f"{self.dataset.owner}/{self.dataset.name}",
            path=dest,
            unzip=False,
            quiet=False,
            force=True,
        )

    @patch("hlm12erc.etl.domain.kaggle_dataset_downloader.kaggle.api.dataset_download_files")
    def test_calls_kaggle_api_is_sensitive_to_force(self, mock_download_files):
        # call the download method with a destination path
        dest = pathlib.Path("test_output")

        self.downloader.download(dest, force=True)
        mock_download_files.assert_called_once_with(
            f"{self.dataset.owner}/{self.dataset.name}",
            path=dest,
            unzip=False,
            quiet=False,
            force=True,
        )

        mock_download_files.reset_mock()
        self.downloader.download(dest, force=False)
        mock_download_files.assert_called_once_with(
            f"{self.dataset.owner}/{self.dataset.name}",
            path=dest,
            unzip=False,
            quiet=False,
            force=False,
        )

    @patch("hlm12erc.etl.domain.kaggle_dataset_downloader.kaggle.api.dataset_download_files")
    def test_raises_exception_if_download_fails(self, mock_download_files):
        # configure the mock to raise an exception
        mock_download_files.side_effect = Exception("Download failed")

        # call the download method with a destination path
        dest = pathlib.Path("test_output")
        with self.assertRaises(Exception):
            self.downloader.download(dest, force=True)

    def tearDown(self):
        # remove any files created during the tests
        for filepath in pathlib.Path("test_output").glob("*"):
            filepath.unlink()
