import logging
import pathlib
from typing import Optional, Union

from .domain.kaggle_dataset import KaggleDataset
from .extraction import KaggleDataExtractor
from .loading import NormalisedDatasetLoader
from .transformation import RawTo1NFTransformer
from .utils import ensure_path

logger = logging.getLogger(__name__)


class ETL:
    """
    Extracts, transforms and loads a dataset from a Kaggle source
    into a 1st Normal Form (1NF) CSV file that can be easily
    consumed by the training process.

    Example:
        >>> from hlm12erc.etl import ETL, KaggleDataset
        >>> dataset = KaggleDataset(owner="hlm12erc", name="kaggle-dataset-downloader")
        >>> ETL(dataset).etl_into(uri_or_folderpath="path/to/1nf/dataset")
    """

    dataset: KaggleDataset
    workspace: pathlib.Path = pathlib.Path("/tmp/hlm12erc/etl")

    def __init__(
        self,
        dataset: KaggleDataset,
        workspace: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        """
        Creates a new ETL pipeline for the given Kaggle dataset,
        responsible for performing the extraction, transformation
        and loading of the dataset from source into a destination.

        :param dataset: The dataset to extract, transform and load.
        :param workspace: The workspace that will be used for processing.
        """
        self.dataset = dataset
        logger.info(f"Kaggle dataset: {dataset.to_kaggle()}")
        if workspace:
            self.workspace = ensure_path(workspace) / "etl"
        logger.info(f"Workspace set to: {workspace}")

    def into(self, uri_or_folderpath: Union[str, pathlib.Path]) -> None:
        """
        Extracts, transforms and loads the dataset into the given destination,
        whereas the destination can be either a local folder, or a Google
        Cloud Storage bucket/folder (for using with Google Collab & TPUs).

        :param uri_or_folderpath: The local folder or Google Cloud Storage bucket/folder to save the dataset to.
        """

        root = self.workspace / self.dataset.to_slug()
        extracted = root / "extracted"
        transformed = root / "transformed"
        loaded = uri_or_folderpath
        logger.info(f"Extracting dataset into: {extracted}")
        KaggleDataExtractor(dataset=self.dataset, workspace=root).extract(dest=extracted)
        logger.info(f"Transforming dataset into: {transformed}")
        RawTo1NFTransformer(src=extracted, workspace=root).transform(dest=transformed)
        logger.info(f"Loading dataset into: {loaded}")
        NormalisedDatasetLoader(src=transformed).load(dest=loaded)
        logger.info("ETL pipeline completed successfully.")
