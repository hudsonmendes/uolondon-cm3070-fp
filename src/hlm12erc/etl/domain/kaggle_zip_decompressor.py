# Python Built-in Modules
import os
import pathlib
import shutil
import zipfile
from typing import Optional

# Third-Party Libraries
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

    def __init__(self, src: pathlib.Path) -> None:
        """
        Creates a new KaggleZipDecompressor.

        :param filepath: The path to the zip file.
        """
        self.filepath = src

    def unpack(self, dest: pathlib.Path, force: bool = False, only_from: Optional[str] = None) -> None:
        """
        Unpacks the dataset to the specified destination folder.

        :param to: The destination folder.
        :param force: If True, the destination folder will be deleted if it already exists.
        :param only_from: The subdirectory from which to extract the files.
        """
        # When forcing AND destination exists, delete it
        self._ensure_destination_is_clean(dest, force)

        # define the subdir to extract
        subdir = f"{only_from.strip('/')}/" if only_from else None

        # if force OR if the destination folder does not exist
        # extracts the data into the destination folder
        with zipfile.ZipFile(self.filepath, "r") as zipfh:
            filenames = zipfh.namelist()
            filecount = len(filenames)
            for filename in tqdm(iterable=filenames, desc="unzip", total=filecount):
                if self._is_filename_from_valid_subdir(filename, subdir):
                    self._extract_file_to_dest(
                        dest=dest,
                        force=force,
                        zipfh=zipfh,
                        filename=filename,
                        subdir=subdir,
                    )

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

    def _is_filename_from_valid_subdir(self, filename: str, subdir: Optional[str]) -> bool:
        """
        Checks if the filename is from the specified subdir.

        :param filename: The filename.
        :param subdir: The subdir.
        :return: True if the filename is from the specified subdir, False otherwise.
        """
        valid_subdir = True
        if subdir:
            valid_subdir = filename.strip("/").startswith(subdir)
        return valid_subdir

    def _extract_file_to_dest(
        self,
        dest: pathlib.Path,
        force: bool,
        zipfh: zipfile.ZipFile,
        filename: str,
        subdir: Optional[str],
    ):
        """
        Extracts the file to the destination folder, stripping
        the subdir from the name, if the file is from the subdir.

        :param dest: The destination folder.
        :param force: If True, the destination folder will be deleted if it already exists.
        :param zipfh: The zipfile handler that is sourciing the files
        :param filename: The filename insiide the zip file
        :param subdir: The subdir from which we are extracting the files, if any
        """
        newfilpath = oldfilepath = dest / filename
        if subdir:
            strippedfilename = filename[len(subdir) :].strip("/")
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
