# Python Built-in Modules
import pathlib


class ETLCommands:
    """
    The CLI commands for Extract, Transform, Load (or "ETL").
    """

    def kaggle(
        self,
        owner: str,
        dataset: str,
        subdir: str,
        dest: pathlib.Path,
        workspace: pathlib.Path | None = None,
        force: bool = False,
    ) -> None:
        """
        Extracts the dataset .zip file into the dest directory, but
        only the files from the dataset's subdirectory.

        :param owner: The owner of the dataset.
        :param dataset: The name of the dataset.
        :param subdir: The subdirectory of the dataset to extract.
        :param workspace: The workspace that will be used for processing.
        :param dest: The destination to extract the dataset into.
        :param force: Whether to force the extraction, even if the destination already exists.
        """
        # My Packages and Modules
        from hlm12erc.etl import ETL, KaggleDataset

        ds = KaggleDataset(owner=owner, name=dataset, subdir=subdir)
        ETL(ds, workspace=workspace).into(uri_or_folderpath=dest, force=force)
