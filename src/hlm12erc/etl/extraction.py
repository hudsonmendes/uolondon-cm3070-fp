# Python Built-in Modules
import logging
import pathlib

# Local Folders
from .domain.kaggle_dataset import KaggleDataset
from .domain.kaggle_dataset_downloader import KaggleDatasetDownloader
from .domain.kaggle_zip_decompressor import KaggleZipDecompressor

logger = logging.getLogger(__name__)


class KaggleDataExtractor:
    """
    Runs the extraction process, from downloading the dataset from
    kaggle to filtering & decompressing it into a temporary directory.

    Example:
        >>> from hlm12erc.etl import KaggleDataExtractor
        >>> extractor = KaggleDataExtractor(dataset=KaggleDataset("hlm12erc", "hlm12erc"), workspace="path/to/workspace")
        >>> extractor.extract(dest="path/to/extracted/dataset/folder")
    """

    dataset: KaggleDataset
    workspace: pathlib.Path

    def __init__(self, dataset: KaggleDataset, workspace: pathlib.Path) -> None:
        """
        Creates a new KaggleDataExtractor.

        :param dataset: The dataset to extract.
        :param workspace: The workspace that will be used for processing.
        """
        self.dataset = dataset
        self.workspace = workspace / "extractor"

    def extract(self, dest: pathlib.Path, force: bool = False) -> None:
        """
        Extracts the dataset .zip file into the dest directory, but
        only the files from the dataset's subdirectory.

        :param dest: The destination to extract the dataset into.
        :param force: Whether to force the extraction, even if the destination already exists.
        """
        logger.info(f"Downloading dataset into: {self.workspace}")
        zipfilepath = KaggleDatasetDownloader(dataset=self.dataset).download(dest=self.workspace, force=force)
        logger.info(f"Extracting dataset from {zipfilepath} into {dest}")
        KaggleZipDecompressor(src=zipfilepath).unpack(only_from=self.dataset.subdir, dest=dest, force=force)
        logger.info("Dataset succesfully extracted")
