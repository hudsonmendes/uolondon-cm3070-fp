# Python Built-in Modules
from typing import List, Tuple

# Third-Party Libraries
import pandas as pd


class MeldTextPreprocessor:
    """
    Preprocessor class for the textual files, creating the dialog prompt
    by concatenating the utterances from the previous dialogues.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def __call__(self, row: pd.Series) -> str:
        dialog_until_now = self._extract_previous_dialogue(dialogue=row.dialogue, before=row.sequence)
        dialog_until_now += [(row.speaker, row.x_text)]
        return self._format_dialog(dialog_until_now)

    def _extract_previous_dialogue(self, dialogue: int, before: int) -> List[Tuple[str, str]]:
        """
        Extracts the dialogue that happened before the current one, based on
        the dialogue number and the sequence number of the current dialogue.

        :param dialogue: The dialogue number of the current dialogue
        :param before: The sequence number of the current dialogue
        :return: The previous dialogue as a list of `MeldDialogueEntry`
        """
        previous_dialogue = self.df[(self.df.dialogue == dialogue) & (self.df.sequence < before)]
        return [(row.speaker, row.x_text) for _, row in previous_dialogue.iterrows()]

    def _format_dialog(self, dialog: List[Tuple[str, str]]) -> str:
        """
        Formats the dialogue as a string, with the speaker's name and the
        utterance, separated by a colon.

        :param dialog: The dialogue to be formatted
        :return: The formatted dialogue
        """
        return "\n\n\n".join([f'The speaker "{speaker}" said:\n###\n{utterance}\n###' for speaker, utterance in dialog])
