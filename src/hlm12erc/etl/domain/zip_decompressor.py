from typing import Optional

import os
import shutil
import pathlib
import zipfile

from tqdm import tqdm


class ZipDecompressor:
    """
    Unpacks a zip file containing a dataset.

    Example:
        >>> from hlm12erc.data import DataUnpacker
        >>> DataUnpacker("meld.zip").only_from("subdir").unpack("/tmp/data/meld", force=True)
    """

    filepath: pathlib.Path
    subdir: Optional[str]

    def __init__(self, src: pathlib.Path) -> None:
        """
        Creates a new DataUnpacker.
        :param filepath: The path to the zip file.
        """
        self.filepath = src
        self.subdir = None

    def only_from(self, subdir: Optional[str]) -> "ZipDecompressor":
        """
        Sets the subdir from which the data should be extracted.
        :param subdir: The subdir.
        """
        self.subdir = f"{subdir.strip('/')}/" if subdir else None
        return self

    def unpack(self, dest: pathlib.Path, force: bool = False) -> None:
        """
        Unpacks the dataset to the specified destination folder.
        :param to: The destination folder.
        :param force: If True, the destination folder will be deleted if it already exists.
        """
        # When forcing AND destination exists, delete it
        self._ensure_destination_is_clean(dest, force)

        # if force OR if the destination folder does not exist
        # extracts the data into the destination folder
        with zipfile.ZipFile(self.filepath, "r") as zipfh:
            filenames = zipfh.namelist()
            filecount = len(filenames)
            for filename in tqdm(iterable=filenames, desc="unzip", total=filecount):
                if self._is_filename_from_valid_subdir(filename):
                    if force or not os.path.exists(dest / filename):
                        zipfh.extract(member=filename, path=dest)

    def _ensure_destination_is_clean(self, to: pathlib.Path, force: bool) -> None:
        """
        Ensures that the destination folder is clean.
        :param to: The destination folder.
        :param force: If True, the destination folder will be deleted if it already exists.
        """
        if force and os.path.exists(to):
            if os.path.isdir(to):
                shutil.rmtree(to, ignore_errors=True)
            else:
                os.remove(to)

    def _is_filename_from_valid_subdir(self, filename: str) -> bool:
        """
        Checks if the filename is from the specified subdir.
        :param filename: The filename.
        :return: True if the filename is from the specified subdir, False otherwise.
        """
        valid_subdir = True
        if self.subdir:
            valid_subdir = filename.strip("/").startswith(self.subdir)
        return valid_subdir
