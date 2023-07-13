import logging
import pathlib
import shutil
from typing import Union
from urllib.parse import urlparse

import google.cloud.storage as gcs

from .utils import ensure_path

logger = logging.getLogger(__name__)


class NormalisedDatasetLoader:
    src: pathlib.Path

    """
    Load a normalised dataset from a source file and save it to a destination
    which maybe a local file or a remote file in a cloud storage.
    """

    def __init__(self, src: pathlib.Path) -> None:
        self.src = src

    def load(self, dest: Union[str, pathlib.Path]) -> None:
        """
        Load the dataset from the source file and save it to the destination.
        :param dest: The destination to save the dataset to.
        """
        logger.info(f"Loading dataset from: {self.src}")

        # verify whether the destination is a local file or a remote file
        # by looking at the prefix of the destination
        loading_locally = isinstance(dest, str) and dest.startswith("gs://")

        # if it is a local file, save the dataset to the local file
        # copying it into the destination path
        if loading_locally:
            logger.info(f"Loading dataset locally into: {dest}")
            destfilepath = ensure_path(dest)
            destfilepath.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.src, destfilepath)
            logger.info(f"Dataset loaded locally into: {destfilepath}")

        # if it is a remote file, save the dataset to the remote file
        # by uploading it to the GCP bucket
        else:
            logger.info(f"Loading dataset remotely into: {dest}")
            desturi = urlparse(str(dest))
            client = gcs.Client()
            bucket_name, blob_name = desturi.hostname, desturi.path[1:]
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(self.src)
            logger.info(f"Dataset loaded remotely into: {desturi.geturl()}")
