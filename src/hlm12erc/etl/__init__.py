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

    Example:
        >>> from hlm12erc.etl import ETL, KaggleDataset
        >>> dataset = KaggleDataset("hlm12erc", "kaggle-dataset-downloader")
        >>> ETL(dataset).etl_into(dest="path/to/1nf/dataset")
    """

    dataset: KaggleDataset
    workspace: pathlib.Path = pathlib.Path("/tmp/hlm12erc/etl")

    def __init__(
        self,
        dataset: KaggleDataset,
        workspace: Optional[Union[str, pathlib.Path]],
    ) -> None:
        """
        Creates a new ETL instance for the given Kaggle dataset.

        :param dataset: The dataset to extract, transform and load.
        :param workspace: The workspace that will be used for processing.
        """
        self.dataset = dataset
        if workspace:
            self.workspace = ensure_path(workspace) / "etl"

    def etl_into(self, dest: Union[str, pathlib.Path]) -> None:
        """
        Extracts, transforms and loads the dataset into the given filepath.

        :param dest: The filepath to load the dataset into.
        """

        root = self.workspace / self.dataset.to_slug()
        extracted = root / "extracted"
        transformed = root / "transformed"
        KaggleDataExtractor(dataset=self.dataset, workspace=root).extract(dest=extracted)
        RawTo1NFTransformer(src=extracted, workspace=root).transform(dest=transformed)
        NormalisedDatasetLoader(src=transformed).load(dest=dest)
