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

    def download(self, dest: pathlib.Path, force: bool) -> pathlib.Path:
        """
        Downloads the dataset into the given destination.

        :param dest: The destination folder to download the dataset into.
        :param force: Whether to force the download, even if the destination already exists.
        :return: The path to the downloaded .zip file.
        """
        kaggle.api.dataset_download_files(
            self.dataset.to_kaggle(),
            path=dest,
            unzip=False,
            quiet=False,
            force=force,
        )
        return dest / f"{self.dataset.name}.zip"
