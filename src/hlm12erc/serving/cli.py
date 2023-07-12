import pathlib
from typing import Optional


class CLI:
    """
    The command line interface (or "CLI") for the HLM12ERC package.
    """

    def erc(self) -> "ERCCommands":
        """
        Return the CLI commands for Emotion Recognition in Conversation (or "ERC").
        """
        return ERCCommands()


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
