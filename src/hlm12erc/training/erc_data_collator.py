# Python Built-in Modules
from typing import List

# Local Folders
from .meld_record import MeldRecord


class ERCDataCollator:
    """
    Collates the data from the ERC dataset into a format that can be used and
    batched by the model, with multiple records turned into lists of its underlying
    datapoints.
    """

    def __call__(self, record: List[MeldRecord]) -> dict:
        """
        Collates the data from the ERC dataset into a format that can be used and
        batched by the model, with multiple records turned into lists of its underlying
        datapoints.

        :param record: The list of records to collate
        :return: The collated data
        """
        return {
            "x_text": [r.to_dialogue_prompt() for r in record],
            "x_visual": [r.visual for r in record],
            "x_audio": [r.audio for r in record],
            "y_true": [r.label for r in record],
        }
