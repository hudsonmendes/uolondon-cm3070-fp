import pathlib
from typing import Optional, Union

from .extraction import KaggleDownloader, ZipDecompressor
from .loading import NormalisedDatasetLoader
from .transformation import RawTo1NFTransformer
from .utils import ensure_path


class ETL:
    """
    Extracts, transforms and loads a dataset from a Kaggle source
    into a 1st Normal Form (1NF) CSV file that can be easily
    consumed by the training process.
    """

    owner: str
    dataset: str
    subdir: Optional[str] = None
    tmpdir: pathlib.Path = pathlib.Path("/tmp/hlm12erc/etl")

    def __init__(self, owner: str, dataset: str) -> None:
        """
        Creates a new ETL instance for the given Kaggle dataset.
        :param owner: The owner of the Kaggle dataset.
        :param dataset: The name of the Kaggle dataset.
        """
        self.owner = owner
        self.dataset = dataset

    def source_only_from(self, subdir: str) -> "ETL":
        """
        Restricts the extraction to only the given subdirectory.
        :param subdir: The subdirectory to extract from.
        :return: The ETL instance.
        """
        self.subdir = subdir
        return self

    def use_tmpdir(self, tmpdir: pathlib.Path) -> "ETL":
        """
        Sets the temporary directory to use for extraction and transformation.
        :param tmpdir: The temporary directory to use.
        :return: The ETL instance.
        """
        self.tmpdir = ensure_path(tmpdir)
        return self

    def etl_into(self, filepath: Union[str, pathlib.Path]) -> None:
        """
        Extracts, transforms and loads the dataset into the given filepath.
        :param filepath: The filepath to load the dataset into.
        """
        tmpzip = self.tmpdir / f"{self.owner}-{self.dataset}.zip"
        tmpdir = self.tmpdir / f"{self.owner}-{self.dataset}"
        tmpcsv = self.tmpdir / f"{self.owner}-{self.dataset}.csv"
        KaggleDownloader(owner=self.owner, dataset=self.dataset).download(dest=tmpzip)
        ZipDecompressor(src=tmpzip).only_from(self.subdir).unpack(dest=tmpdir, force=True)
        RawTo1NFTransformer(src=tmpdir).transform(dest=tmpcsv)
        NormalisedDatasetLoader(src=tmpcsv).load(dest=filepath)
