import pathlib
from typing import Optional

from hlm12erc.etl import ETL, KaggleDataset


class CLI:
    """
    The command line interface (or "CLI") for the HLM12ERC package.
    """

    def etl(self) -> "ETLCommands":
        """
        Return the CLI commands for Extract, Transform, Load (or "ETL").
        """
        return ETLCommands()

    def erc(self) -> "ERCCommands":
        """
        Return the CLI commands for Emotion Recognition in Conversation (or "ERC").
        """
        return ERCCommands()


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
        workspace: Optional[pathlib.Path] = None,
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
        ds = KaggleDataset(owner=owner, name=dataset, subdir=subdir)
        ETL(ds, workspace=workspace).into(uri_or_folderpath=dest, force=force)


class ERCCommands:
    """
    The CLI commands for Emotion Recognition in Conversation (or "ERC").
    """

    def classify_emotion(
        self,
        audio: pathlib.Path,
        video: pathlib.Path,
        dialog: pathlib.Path,
        utterance: str,
        out: Optional[pathlib.Path] = None,
    ) -> None:
        """
        Classify the emotion of the utterance in the dialog.
        :param audio: The path to the audio file.
        :param video: The path to the video file.
        :param dialog: The path to the dialog file.
        :param utterance: The utterance to classify the emotion of.
        :param out: If None, outputs to stdout, otherwise outputs to the file.
        """
        assert audio and audio.exists()
        assert video and video.exists()
        assert dialog and dialog.exists()
        assert utterance and isinstance(utterance, str)
        assert out is None or isinstance(out, pathlib.Path)
        raise NotImplementedError("Not yet implemented")
