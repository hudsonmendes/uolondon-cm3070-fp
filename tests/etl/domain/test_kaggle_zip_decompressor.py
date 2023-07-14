# Python Built-in Modules
import os
import pathlib
import shutil
import unittest
import zipfile

# My Packages and Modules
from hlm12erc.etl.domain.kaggle_zip_decompressor import KaggleZipDecompressor


class TestKaggleZipDecompressor(unittest.TestCase):
    def setUp(self):
        self.src_path = pathlib.Path("/tmp/hlm12erc/tests/test_data/")
        self.dest_path = pathlib.Path("/tmp/hlm12erc/tests/test_output/")
        self.zipfile_path = self.src_path / "test_dataset.zip"
        self.zipfile_path.parent.mkdir(parents=True, exist_ok=True)
        self.decompressor = KaggleZipDecompressor(self.zipfile_path)

        # create a dummy zip file for use during the test
        self.zipfile = zipfile.ZipFile(self.zipfile_path, "w")
        self.zipfile.writestr("file1.txt", "test data")
        self.zipfile.writestr("subdir/file2.txt", "test data")
        self.zipfile.close()

    def tearDown(self):
        # remove any files or folders created during the tests
        shutil.rmtree(self.src_path, ignore_errors=True)
        shutil.rmtree(self.dest_path, ignore_errors=True)

    def test_calls_extract_with_correct_arguments(self):
        # call the unpack method with a destination path
        self.decompressor.unpack(self.dest_path)

        # check that the files were extracted to the correct destination path
        self.assertTrue((self.dest_path / "file1.txt").exists())
        self.assertTrue((self.dest_path / "subdir/file2.txt").exists())

    def test_does_not_extract_files_from_invalid_subdir(self):
        # call the unpack method with a destination path and a subdir
        self.decompressor.unpack(dest=self.dest_path, only_from="subdir")

        # check that only the valid file was extracted
        self.assertFalse((self.dest_path / "file1.txt").exists())
        self.assertFalse((self.dest_path / "subdir/file2.txt").exists())
        self.assertTrue((self.dest_path / "file2.txt").exists())

    def test_deletes_destination_folder_if_force_is_true(self):
        # create a destination folder and a file inside it
        os.makedirs(self.dest_path, exist_ok=True)
        (self.dest_path / "dirt.txt").touch()

        # call the unpack method with a destination path and force=True
        self.decompressor.unpack(dest=self.dest_path, force=True)

        # check that the destination folder was deleted
        self.assertFalse((self.dest_path / "dirt.txt").exists())

    def test_does_not_delete_destination_folder_if_force_is_false(self):
        # create a destination folder and a file inside it
        os.makedirs(self.dest_path, exist_ok=True)
        (self.dest_path / "dirt.txt").touch()

        # call the unpack method with a destination path and force=False
        self.decompressor.unpack(dest=self.dest_path, force=False)

        # check that the destination folder was not deleted
        self.assertTrue((self.dest_path / "dirt.txt").exists())
