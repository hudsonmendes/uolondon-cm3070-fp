# Python Built-in Modules
import pathlib
import tempfile
import unittest
from unittest.mock import patch

# My Packages and Modules
from hlm12erc.etl import KaggleDataExtractor, KaggleDataset


class TestKaggleDataExtractor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset = KaggleDataset("hlm12erc", "hlm12erc")
        self.extractor = KaggleDataExtractor(dataset=self.dataset, workspace=pathlib.Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("hlm12erc.etl.domain.kaggle_dataset_downloader.KaggleDatasetDownloader.download")
    @patch("hlm12erc.etl.domain.kaggle_zip_decompressor.KaggleZipDecompressor.unpack")
    def test_extract_calls_download_and_unpack(self, mock_unpack, mock_download):
        # Arrange
        dest = pathlib.Path(self.temp_dir.name) / "extracted"
        mock_download.return_value = pathlib.Path(self.temp_dir.name) / "dataset.zip"

        # Act
        self.extractor.extract(dest=dest)

        # Assert
        mock_download.assert_called_once_with(dest=self.extractor.workspace, force=False)
        mock_unpack.assert_called_once_with(only_from=self.dataset.subdir, dest=dest, force=False)

    @patch("hlm12erc.etl.domain.kaggle_dataset_downloader.KaggleDatasetDownloader.download")
    @patch("hlm12erc.etl.domain.kaggle_zip_decompressor.KaggleZipDecompressor.unpack")
    def test_extract_calls_download_and_unpack_with_force(self, mock_unpack, mock_download):
        # Arrange
        dest = pathlib.Path(self.temp_dir.name) / "extracted"
        mock_download.return_value = pathlib.Path(self.temp_dir.name) / "dataset.zip"

        # Act
        self.extractor.extract(dest=dest, force=True)

        # Assert
        mock_download.assert_called_once_with(dest=self.extractor.workspace, force=True)
        mock_unpack.assert_called_once_with(only_from=self.dataset.subdir, dest=dest, force=True)

    @patch("hlm12erc.etl.domain.kaggle_dataset_downloader.KaggleDatasetDownloader.download")
    def test_extract_raises_exception_if_download_fails(self, mock_download):
        # Arrange
        dest = pathlib.Path(self.temp_dir.name) / "extracted"
        mock_download.side_effect = Exception("Download failed")

        # Act & Assert
        with self.assertRaises(Exception):
            self.extractor.extract(dest=dest)

    @patch("hlm12erc.etl.domain.kaggle_dataset_downloader.KaggleDatasetDownloader.download")
    @patch("hlm12erc.etl.domain.kaggle_zip_decompressor.KaggleZipDecompressor.unpack")
    def test_extract_raises_exception_if_unpack_fails(self, mock_unpack, mock_download):
        # Arrange
        dest = pathlib.Path(self.temp_dir.name) / "extracted"
        mock_download.return_value = pathlib.Path(self.temp_dir.name) / "dataset.zip"
        mock_unpack.side_effect = Exception("Unpack failed")

        # Act & Assert
        with self.assertRaises(Exception):
            self.extractor.extract(dest=dest)
