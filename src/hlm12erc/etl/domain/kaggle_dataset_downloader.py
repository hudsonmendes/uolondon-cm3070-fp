import pathlib

import kaggle

from .kaggle_dataset import KaggleDataset


class KaggleDatasetDownloader:
    dataset: KaggleDataset

    """
    Downloads a dataset (.zip) from a Kaggle Source.

    Example:
        >>> from hlm12erc.data import DatasetDownloader
        >>> DatasetDownloader("hlm12erc", "meld").download("/tmp/data/meld.zip")
    """

    def __init__(self, dataset: KaggleDataset) -> None:
        """
        Creates a new KaggleDatasetDownloader.
        :param owner: The owner of the dataset.
        :param dataset: The name of the dataset.
        """
        self.dataset = dataset

    def download(self, dest: pathlib.Path) -> None:
        """
        Downloads the dataset to the specified destination folder.
        :param to: The destination folder.
        """
        kaggle.api.dataset_download_files(
            f"{self.dataset.owner}/{self.dataset.name}",
            path=dest,
            unzip=False,
        )
