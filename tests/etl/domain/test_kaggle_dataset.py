# Python Built-in Modules
import unittest


class TestKaggleDataset(unittest.TestCase):
    def test_returns_slugified_name(self):
        # create a dataset with a name that needs to be slugified
        dataset = self._new_subject("hlm12erc", "kaggle-dataset-downloader")

        # call the to_slug method and check that the returned value is correct
        expected_slug = "hlm12erc-kaggle-dataset-downloader"
        self.assertEqual(dataset.to_slug(), expected_slug)

    def test_returns_slugified_name_without_subdirectory(self):
        # create a dataset with a name that needs to be slugified and a subdirectory
        dataset = self._new_subject("hlm12erc", "kaggle-dataset-downloader", "data")

        # call the to_slug method and check that the returned value is correct
        expected_slug = "hlm12erc-kaggle-dataset-downloader"
        self.assertEqual(dataset.to_slug(), expected_slug)

    def test_returns_slugified_name_with_uppercase_characters(self):
        # create a dataset with a name that contains uppercase characters
        dataset = self._new_subject("hlm12erc", "KaggleDatasetDownloader")

        # call the to_slug method and check that the returned value is correct
        expected_slug = "hlm12erc-kaggledatasetdownloader"
        self.assertEqual(dataset.to_slug(), expected_slug)

    def test_returns_slugified_name_with_special_characters(self):
        # create a dataset with a name that contains special characters
        dataset = self._new_subject("hlm12erc", "kaggle_dataset_downloader")

        # call the to_slug method and check that the returned value is correct
        expected_slug = "hlm12erc-kaggle-dataset-downloader"
        self.assertEqual(dataset.to_slug(), expected_slug)

    def test_to_kaggle_returns_kaggle_path(self):
        # create a dataset with a name and a subdirectory
        dataset = self._new_subject("hlm12erc", "kaggle-dataset-downloader", "data")

        # asserts that the kaggle path is correct
        self.assertEqual(dataset.to_kaggle(), "hlm12erc/kaggle-dataset-downloader")

    def _new_subject(self, owner: str, name: str, subdir: str = None):
        # local import to avoid etl dependencies becoming global requirements
        # My Packages and Modules
        from hlm12erc.etl.domain.kaggle_dataset import KaggleDataset

        # create and return an instance of the subject class
        return KaggleDataset(owner, name, subdir)
