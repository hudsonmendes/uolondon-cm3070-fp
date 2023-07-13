import os
import pathlib
import shutil
import unittest
import zipfile

from hlm12erc.etl import ZipDecompressor


class TestZipDecompressor(unittest.TestCase):
    def setUp(self):
        self.test_dir = pathlib.Path("/tmp/test_data")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_unpack(self):
        # create a zip file with some data
        zip_path = self.test_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zipfh:
            zipfh.writestr("file1.txt", "file1 contents")
            zipfh.writestr("subdir/file2.txt", "file2 contents")

        # unpack the zip file
        unpacker = ZipDecompressor(zip_path)
        unpacker.unpack(self.test_dir / "unpacked")

        # check that the files were unpacked correctly
        self.assertTrue((self.test_dir / "unpacked/file1.txt").exists())
        self.assertTrue((self.test_dir / "unpacked/subdir/file2.txt").exists())

    def test_unpack_only_from(self):
        # create a zip file with some data
        zip_path = self.test_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zipfh:
            zipfh.writestr("file1.txt", "file1 contents")
            zipfh.writestr("subdir/file2.txt", "file2 contents")

        # unpack only the files from the "subdir" directory
        unpacker = ZipDecompressor(zip_path).only_from("subdir")
        unpacker.unpack(self.test_dir / "unpacked")

        # check that only the files from the "subdir" directory were unpacked
        self.assertFalse((self.test_dir / "unpacked/file1.txt").exists())
        self.assertTrue((self.test_dir / "unpacked/subdir/file2.txt").exists())

    def test_unpack_force_is_true(self):
        # create a zip file with some data
        zip_path = self.test_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zipfh:
            zipfh.writestr("file1.txt", "file1 contents")

        # create a file with the same name as the destination folder
        os.makedirs(self.test_dir / "unpacked", exist_ok=True)
        (self.test_dir / "unpacked/dirt.txt").touch()
        self.assertTrue((self.test_dir / "unpacked/dirt.txt").exists())
        self.assertFalse((self.test_dir / "unpacked/file1.txt").exists())

        # unpack the zip file with force=True
        unpacker = ZipDecompressor(zip_path)
        unpacker.unpack(self.test_dir / "unpacked", force=True)

        # check that the destination folder was deleted and the files were unpacked correctly
        self.assertFalse((self.test_dir / "unpacked/dirt.txt").exists())
        self.assertTrue((self.test_dir / "unpacked/file1.txt").exists())

    def test_unpack_force_is_false(self):
        # create a zip file with some data
        zip_path = self.test_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zipfh:
            zipfh.writestr("file1.txt", "file1 contents")

        # create a file with the same name as the destination folder
        os.makedirs(self.test_dir / "unpacked", exist_ok=True)
        (self.test_dir / "unpacked/dirt.txt").touch()
        self.assertTrue((self.test_dir / "unpacked/dirt.txt").exists())
        self.assertFalse((self.test_dir / "unpacked/file1.txt").exists())

        # unpack the zip file with force=False
        unpacker = ZipDecompressor(zip_path)
        unpacker.unpack(self.test_dir / "unpacked", force=False)

        # check that the destination folder was not deleted and the files were unpacked correctly
        self.assertTrue((self.test_dir / "unpacked/dirt.txt").exists())
        self.assertTrue((self.test_dir / "unpacked/file1.txt").exists())

    def test_unpack_only_from_subdir(self):
        # create a zip file with some data
        zip_path = self.test_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zipfh:
            zipfh.writestr("dir1/file1.txt", "file1 contents")
            zipfh.writestr("dir2/file1.txt", "file1 contents")

        # unpack the zip file with force=True
        unpacker = ZipDecompressor(zip_path)
        unpacker.only_from(subdir="dir2")
        unpacker.unpack(self.test_dir / "unpacked")

        # check that the files were unpacked correctly
        self.assertFalse((self.test_dir / "unpacked/dir1/file1.txt").exists())
        self.assertTrue((self.test_dir / "unpacked/dir2/file1.txt").exists())

    def test_unpack_from_all_subdir(self):
        # create a zip file with some data
        zip_path = self.test_dir / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zipfh:
            zipfh.writestr("dir1/file1.txt", "file1 contents")
            zipfh.writestr("dir2/file1.txt", "file1 contents")

        # unpack the zip file with force=True
        unpacker = ZipDecompressor(zip_path)
        unpacker.only_from(subdir=None)
        unpacker.unpack(self.test_dir / "unpacked")

        # check that the files were unpacked correctly
        self.assertTrue((self.test_dir / "unpacked/dir1/file1.txt").exists())
        self.assertTrue((self.test_dir / "unpacked/dir2/file1.txt").exists())
