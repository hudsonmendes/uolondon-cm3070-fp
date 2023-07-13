import pathlib

import kaggle

from .kaggle_dataset import KaggleDataset


class KaggleDatasetDownloader:
    """
    Downloads a Kaggle dataset into a destination .zip file.

    Example:
        >>> from hlm12erc.etl import KaggleDataset, KaggleDatasetDownloader
        >>> dataset = KaggleDataset("hlm12erc", "kaggle-dataset-downloader")
        >>> zip_filepath = KaggleDatasetDownloader(dataset).download()
    """

    dataset: KaggleDataset

    def __init__(self, dataset: KaggleDataset) -> None:
        """
        Creates a new KaggleDatasetDownloader.

        :param dataset: The dataset to download.
        """
        self.dataset = dataset

    def download(self, dest: pathlib.Path) -> None:
        """
        Downloads the dataset into the given destination.

        :param dest: The destination to download the dataset into.
        """
        kaggle.api.dataset_download_files(
            f"{self.dataset.owner}/{self.dataset.name}",
            path=dest,
            unzip=False,
        )
