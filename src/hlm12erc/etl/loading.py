# Python Built-in Modules
import logging
import os
import pathlib
import shutil
from typing import Callable, Union
from urllib.parse import urlparse

# Third-Party Libraries
import google.cloud.storage as gcs
from tqdm import tqdm, trange

# Local Folders
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

    def load(self, dest: Union[str, pathlib.Path], force: bool) -> None:
        """
        Load the dataset from the source file and save it to the destination.
        :param dest: The destination to save the dataset to.
        :param force: If True, forces loading even if location already has data
        """
        logger.info(f"Loading dataset from: {self.src}")

        # verify whether the destination is a local file or a remote file
        # by looking at the prefix of the destination
        loading_locally = not (isinstance(dest, str) and "://" in dest)

        # if it is a local file, save the dataset to the local file
        # copying it into the destination path
        if loading_locally:
            self._load_into_filesystem(dest, force=force)

        # if it is a remote file, save the dataset to the remote file
        # by uploading it to the GCP bucket
        else:
            self._load_into_google_storage(dest, force=force)

    def _load_into_filesystem(self, dest: Union[str, pathlib.Path], force: bool):
        logger.info(f"Loading dataset locally into: {dest}")
        destfolderpath = ensure_path(dest)
        destfolderpath.parent.mkdir(parents=True, exist_ok=True)
        filecount = len(list(self.src.glob("*")))
        with trange(0, filecount, desc="copying data") as pbar:
            copy_fn = self._copy_with_progressbar_fn_builder(pbar, force=force)
            shutil.copytree(self.src, destfolderpath, dirs_exist_ok=True, copy_function=copy_fn)
            logger.info(f"Dataset loaded locally into: {destfolderpath}")

    def _load_into_google_storage(self, dest: Union[str, pathlib.Path], force: bool):
        logger.info(f"Loading dataset remotely into: {dest}")
        desturi = urlparse(str(dest))
        client = gcs.Client()
        bucket_name, blob_name = desturi.hostname, desturi.path[1:]
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(self.src)
        logger.info(f"Dataset loaded remotely into: {desturi.geturl()}")

    def _copy_with_progressbar_fn_builder(self, pbar: tqdm, force: bool) -> Callable[[str, str], None]:
        """
        Creates a function that copies a single files (from src to destination)
        but that also updates a progressbar when it does

        :param pbar: the TQDM progress bar that will be updated by 1
        :param force: if True, copies even if the destination already contains a file
        :returns: a function that takes in 2 strings (src, dest) and copies the files
        """

        def copy_fn(s, d):
            if force or not os.path.exists(d):
                shutil.copy2(s, d)
            pbar.update(1)

        return copy_fn
