# Python Built-in Modules
from typing import List

# My Packages and Modules
from hlm12erc.modelling import ERCLabelEncoder

# Local Folders
from .meld_record import MeldRecord


class ERCDataCollator:
    """
    Collates the data from the ERC dataset into a format that can be used and
    batched by the model, with multiple records turned into lists of its underlying
    datapoints.
    """

    def __init__(self, label_encoder: ERCLabelEncoder) -> None:
        """
        Initialise the ERCDataCollator class with the given ERCLabelEncoder object.

        :param label_encoder: ERCLabelEncoder object containing the label encoder
        """
        self.label_encoder = label_encoder

    def __call__(self, record: List[MeldRecord]) -> dict:
        """
        Collates the data from the ERC dataset into a format that can be used and
        batched by the model, with multiple records turned into lists of its underlying
        datapoints. We also encode the labels to make it easier to use in the model.

        :param record: The list of records to collate
        :return: The collated data
        """
        return {
            "x_text": [r.to_dialogue_prompt() for r in record],
            "x_visual": [r.visual for r in record],
            "x_audio": [r.audio for r in record],
            "y_true": self.label_encoder([r.label for r in record]),
        }
