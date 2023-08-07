# Python Built-in Modules
import pathlib
import shutil
import unittest
from unittest.mock import patch


class TestNormalisedDatasetLoader(unittest.TestCase):
    def setUp(self):
        # local import to avoid etl dependencies becoming global requirements
        # My Packages and Modules
        from hlm12erc.etl.loading import NormalisedDatasetLoader

        self.src = pathlib.Path("/tmp/hlm12erc/tests/test_data/")
        self.src.mkdir(parents=True, exist_ok=True)
        (self.src / "file1.csv").touch()
        (self.src / "file2.csv").touch()
        self.loader = NormalisedDatasetLoader(self.src)

    def tearDown(self):
        del self.loader
        shutil.rmtree(self.src)

    @patch("hlm12erc.etl.loading.NormalisedDatasetLoader._load_into_filesystem")
    def test_load_local(self, mock_load_into_filesystem):
        dest = pathlib.Path("/tmp/hlm12erc/tests/test_output/")
        dest.mkdir(parents=True, exist_ok=True)
        try:
            force = False
            self.loader.load(dest, force)
            mock_load_into_filesystem.assert_called_once_with(dest, force=force)
        finally:
            shutil.rmtree(dest)

    @patch("hlm12erc.etl.loading.NormalisedDatasetLoader._load_into_google_storage")
    def test_load_remote(self, mock_load_into_google_storage):
        dest = "gs://bucket/path/to/dest"
        force = False
        self.loader.load(dest, force)
        mock_load_into_google_storage.assert_called_once_with(dest, force=force)

    @patch("hlm12erc.etl.loading.NormalisedDatasetLoader._load_into_filesystem")
    def test_load_local_force(self, mock_load_into_filesystem):
        dest = pathlib.Path("/tmp/hlm12erc/tests/test_output/")
        dest.mkdir(parents=True, exist_ok=True)
        try:
            force = True
            self.loader.load(dest, force)
            mock_load_into_filesystem.assert_called_once_with(dest, force=force)
        finally:
            shutil.rmtree(dest)

    @patch("hlm12erc.etl.loading.NormalisedDatasetLoader._load_into_google_storage")
    def test_load_remote_force(self, mock_load_into_google_storage):
        dest = "gs://bucket/path/to/dest"
        force = True
        self.loader.load(dest, force)
        mock_load_into_google_storage.assert_called_once_with(dest, force=force)

    @patch("hlm12erc.etl.loading.NormalisedDatasetLoader._load_into_filesystem")
    def test_load_local_exception(self, mock_load_into_filesystem):
        dest = pathlib.Path("/tmp/hlm12erc/tests/test_output/")
        dest.mkdir(parents=True, exist_ok=True)
        try:
            force = False
            mock_load_into_filesystem.side_effect = Exception("Error")
            with self.assertRaises(Exception):
                self.loader.load(dest, force)
        finally:
            shutil.rmtree(dest)

    @patch("hlm12erc.etl.loading.NormalisedDatasetLoader._load_into_google_storage")
    def test_load_remote_exception(self, mock_load_into_google_storage):
        dest = "gs://bucket/path/to/dest"
        force = False
        mock_load_into_google_storage.side_effect = Exception("Error")
        with self.assertRaises(Exception):
            self.loader.load(dest, force)
