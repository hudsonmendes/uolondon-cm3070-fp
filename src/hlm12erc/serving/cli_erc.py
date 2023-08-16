# Python Built-in Modules
import pathlib


class ERCCommands:
    """
    The CLI commands for Emotion Recognition in Conversation (or "ERC").
    """

    def train(
        self,
        config: pathlib.Path,
        train_dataset: pathlib.Path | str,
        valid_dataset: pathlib.Path | str,
        n_epochs: int,
        batch_size: int,
        out: pathlib.Path | str | None = None,
    ) -> None:
        """
        Trains the emotion recognition classifier on the train and validation datasets.

        :param config: The path to the config file, usually a file within `./config`.
        :param train_dataset: The path to the train dataset.
        :param valid_dataset: The path to the validation dataset.
        :param n_epochs: The number of epochs to train for.
        :param batch_size: The batch size to use for training.
        :param out: The path to save the model to, defaults to './target'
        """
        # My Packages and Modules
        from hlm12erc.modelling import ERCConfigLoader
        from hlm12erc.training import ERCTrainer, MeldDataset

        train_dataset = pathlib.Path(train_dataset) if isinstance(train_dataset, str) else train_dataset
        valid_dataset = pathlib.Path(valid_dataset) if isinstance(valid_dataset, str) else valid_dataset
        out = pathlib.Path(out) if isinstance(out, str) and out is not None else out
        config = pathlib.Path(config) if isinstance(config, str) else config
        ERCTrainer(ERCConfigLoader(config).load()).train(
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
        out: pathlib.Path | None = None,
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
