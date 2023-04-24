import pathlib


class ERCCommands:
    def classify_emotion(
        self,
        audio: pathlib.Path,
        video: pathlib.Path,
        dialog: pathlib.Path,
        utterance: str,
    ):
        assert audio and audio.exists()
        assert video and video.exists()
        assert dialog and dialog.exists()
        assert utterance and isinstance(utterance, str)
        raise NotImplementedError("Not yet implemented")
