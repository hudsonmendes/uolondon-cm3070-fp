# Python Built-in Modules
import pathlib
import wave
from typing import Optional

# Third-Party Libraries
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import trange

# Local Folders
from .meld_record import MeldRecord
from .meld_record_preprocessor_audio import MeldAudioPreprocessor
from .meld_record_preprocessor_text import MeldTextPreprocessor
from .meld_record_preprocessor_visual import MeldVisualPreprocessor


class MeldRecordReader:
    def __init__(
        self,
        filename: str,
        filedir: pathlib.Path,
        df: pd.DataFrame,
        device: Optional[torch.device] = None,
    ):
        self.df = df
        self.filename = filename
        self.filedir = filedir
        self.preprocessor_text = MeldTextPreprocessor(df=df)
        self.preprocessor_visual = MeldVisualPreprocessor(device=device)
        self.preprocessor_audio = MeldAudioPreprocessor(device=device)

    def read_all_valid(self) -> list[MeldRecord]:
        records = [self.read_at(i) for i in trange(len(self.df), desc=f"meld({self.filename})")]
        return [record for record in records if record is not None]

    def read_at(self, i: int) -> Optional[MeldRecord]:
        if i < len(self.df):
            row = self.df.iloc[i]

            with (
                Image.open(str(self.filedir / row.x_visual)) as image_file,
                wave.open(str(self.filedir / row.x_audio)) as audio_file,
            ):
                text = self.preprocessor_text(row)
                visual = self.preprocessor_visual(image_file)
                audio = self.preprocessor_audio(audio_file)
                assert visual.shape is not None and visual.device == self.preprocessor_visual.device
                assert audio.shape is not None and audio.device == self.preprocessor_audio.device
                return MeldRecord(text=text, visual=visual, audio=audio, label=row.label)
        else:
            return None
