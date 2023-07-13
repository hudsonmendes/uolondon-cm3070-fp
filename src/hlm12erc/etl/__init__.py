import pathlib
from typing import Optional, Union

from .domain.kaggle_dataset import KaggleDataset
from .extraction import KaggleDataExtractor
from .loading import NormalisedDatasetLoader
from .transformation import RawTo1NFTransformer
from .utils import ensure_path


class ETL:
    """
    Extracts, transforms and loads a dataset from a Kaggle source
    into a 1st Normal Form (1NF) CSV file that can be easily
    consumed by the training process.
    """

    dataset: KaggleDataset
    workspace: pathlib.Path = pathlib.Path("/tmp/hlm12erc/etl")

    def __init__(self, dataset) -> None:
        """
        Creates a new ETL instance for the given Kaggle dataset.
        :param owner: The owner of the Kaggle dataset.
        :param dataset: The name of the Kaggle dataset.
        """
        self.dataset = dataset

    def use_workspace(self, workspace: pathlib.Path) -> "ETL":
        """
        Sets the temporary directory to use for extraction and transformation.
        :param tmpdir: The temporary directory to use.
        :return: The ETL instance.
        """
        if workspace:
            self.workspace = ensure_path(workspace)
        return self

    def etl_into(self, dest: Union[str, pathlib.Path]) -> None:
        """
        Extracts, transforms and loads the dataset into the given filepath.
        :param dest: The filepath to load the dataset into.
        """

        root = self.workspace / f"{self.owner}-{self.dataset}"
        extracted = KaggleDataExtractor(dataset=self.dataset, workspace=root).extract()
        tranformed = RawTo1NFTransformer(src=extracted, workspace=root).transform()
        NormalisedDatasetLoader(src=tranformed).load(dest=dest)
