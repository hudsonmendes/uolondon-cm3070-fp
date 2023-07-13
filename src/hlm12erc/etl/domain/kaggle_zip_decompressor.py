import os
import pathlib
import shutil
import zipfile
from typing import Optional

from tqdm import tqdm


class KaggleZipDecompressor:
    """
    Unpacks a zip file containing a dataset.

    Example:
        >>> from hlm12erc.etl import KaggleZipDecompressor
        >>> decompressor = KaggleZipDecompressor(src="path/to/dataset.zip")
        >>> decompressor.only_from("subdir")
        >>> decompressor.unpack(dest="path/to/extracted/dataset")
    """

    filepath: pathlib.Path
    subdir: Optional[str]

    def __init__(self, src: pathlib.Path) -> None:
        """
        Creates a new KaggleZipDecompressor.

        :param filepath: The path to the zip file.
        """
        self.filepath = src
        self.subdir = None

    def only_from(self, subdir: Optional[str]) -> "KaggleZipDecompressor":
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
                    self._extract_file_to_dest(dest, force, zipfh, filename)

        # clean empty folders within the destination folder
        self._clean_empty_folders(dest)

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

    def _extract_file_to_dest(self, dest, force, zipfh, filename):
        """
        Extracts the file to the destination folder, stripping
        the subdir from the name, if the file is from the subdir.

        :param dest: The destination folder.
        :param force: If True, the destination folder will be deleted if it already exists.
        :param zipfh: The zipfile handler.
        :param filename: The filename.
        """
        newfilpath = oldfilepath = dest / filename
        if self.subdir:
            strippedfilename = filename[len(self.subdir) :].strip("/")
            newfilpath = dest / strippedfilename
        if force or not newfilpath.exists():
            zipfh.extract(member=filename, path=dest)
            if oldfilepath != newfilpath:
                newfilpath.parent.mkdir(parents=True, exist_ok=True)
                oldfilepath.rename(newfilpath)

    def _clean_empty_folders(self, dest: pathlib.Path) -> None:
        """
        Scans through all folders inside `dest` and removes folders that do not contain files.

        :param dest: The destination folder.
        """
        for root, dirs, files in os.walk(dest, topdown=False):
            for dir in dirs:
                dirpath = os.path.join(root, dir)
                if not os.listdir(dirpath):
                    os.rmdir(dirpath)
