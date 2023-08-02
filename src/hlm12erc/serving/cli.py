# Python Built-in Modules
import pathlib
from typing import Optional


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
        # My Packages and Modules
        from hlm12erc.etl import ETL, KaggleDataset

        ds = KaggleDataset(owner=owner, name=dataset, subdir=subdir)
        ETL(ds, workspace=workspace).into(uri_or_folderpath=dest, force=force)


class ERCCommands:
    """
    The CLI commands for Emotion Recognition in Conversation (or "ERC").
    """

    def train(
        self,
        train_dataset: pathlib.Path,
        valid_dataset: pathlib.Path,
        n_epochs: int,
        batch_size: int = 16,
        config: Optional[pathlib.Path] = None,
        out: Optional[pathlib.Path] = None,
    ) -> None:
        """
        Trains the emotion recognition classifier on the train and validation datasets.

        :param train_dataset: The path to the train dataset.
        :param valid_dataset: The path to the validation dataset.
        :param n_epochs: The number of epochs to train for.
        :param batch_size: The batch size to use for training, defaults to 16.
        :param config: The path to the config file, defaults to the default settings
        :param out: The path to save the model to, defaults to './target'
        """

        # My Packages and Modules
        from hlm12erc.modelling import ERCConfig
        from hlm12erc.training import ERCConfigLoader, ERCTrainer, MeldDataset

        train_dataset = pathlib.Path(train_dataset) if not isinstance(train_dataset, pathlib.Path) else train_dataset
        valid_dataset = pathlib.Path(valid_dataset) if not isinstance(valid_dataset, pathlib.Path) else valid_dataset
        out = pathlib.Path(out) if out is not None and not isinstance(out, pathlib.Path) else out
        config = pathlib.Path(config) if config is not None and not isinstance(config, pathlib.Path) else config
        effective_config = ERCConfigLoader(config).load() if config else ERCConfig()
        ERCTrainer(effective_config).train(
            data=(MeldDataset(train_dataset), MeldDataset(valid_dataset)),
            batch_size=batch_size,
            n_epochs=n_epochs,
            save_to=(out or pathlib.Path("./target")),
        )

    def classify(
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
