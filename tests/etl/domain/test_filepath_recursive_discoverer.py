# Python Built-in Modules
import pathlib
import shutil
import unittest

# My Packages and Modules
from hlm12erc.etl.domain.filepath_recursive_discoverer import FilepathRecursiveDiscoverer


class TestFilepathRecursiveDiscoverer(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path("/tmp/hlm12erc/test_data")
        self.discoverer = FilepathRecursiveDiscoverer(self.root)
        (self.root / "subdir1").mkdir(parents=True, exist_ok=True)
        (self.root / "file.txt").touch(exist_ok=True)
        (self.root / "subdir1/file.txt").touch(exist_ok=True)
        (self.root / "subdir1/another-file.txt").touch(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_returns_path_if_file_exists(self):
        # call the __call__ method with an existing filename
        path = self.discoverer("file.txt")

        # check that the returned path exists and is a file
        self.assertTrue(path.exists())
        self.assertTrue(path.is_file())

    def test_returns_first_path_found_if_file_exists(self):
        # call the __call__ method with an existing filename
        path = self.discoverer("file.txt")

        # check that the path of the file is the topmost one
        self.assertEqual(path, self.root / "file.txt")

    def test_raises_error_if_file_does_not_exist(self):
        # call the __call__ method with a non-existing filename
        with self.assertRaises(FileNotFoundError):
            self.discoverer("non_existing_file.txt")

    def test_returns_first_matching_path(self):
        # call the __call__ method with a filename that matches multiple files
        path = self.discoverer("another-file.txt")

        # check that the returned path is the first matching path
        self.assertEqual(path, self.root / "subdir1" / "another-file.txt")
