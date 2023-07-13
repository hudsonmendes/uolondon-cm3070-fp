from typing import Optional

import pathlib

from .domain.kaggle_dataset import KaggleDataset
from .domain.kaggle_dataset_downloader import KaggleDatasetDownloader
from .domain.zip_decompressor import ZipDecompressor


class KaggleDataExtractor:
    """
    Runs the extraction process, from downloading the dataset from
    kaggle to filtering & decompressing it into a temporary directory.
    """

    dataset: KaggleDataset
    workspace: pathlib.Path = pathlib.Path("/tmp/hlm12erc/etl/extraction")

    def __init__(self, dataset: KaggleDataset, workspace: Optional[pathlib.Path]) -> None:
        """
        Creates a new KaggleDataExtractor.
        :param tmpdir: The temporary directory to use.
        :param owner: The owner of the dataset.
        :param dataset: The name of the dataset.
        """
        self.dataset = dataset
        if workspace:
            self.workspace = workspace

    def extract(self) -> pathlib.Path:
        zip = self.workspace / f"{self.dataset.owner}-{self.dataset.name}.zip"
        dest = self.workspace / "extracted"
        KaggleDatasetDownloader(dataset=self.dataset).download(dest=zip)
        ZipDecompressor(src=zip).only_from(self.dataset.subdir).unpack(dest=dest, force=True)
        return dest
