# Python Built-in Modules
from typing import List

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling import ERCConfig, ERCLabelEncoder

# Local Folders
from .meld_record import MeldRecord


class ERCDataCollator:
    """
    Collates the data from the ERC dataset into a format that can be used and
    batched by the model, with multiple records turned into lists of its underlying
    datapoints.
    """

    LABEL_NAME: str = "y_true"

    config: ERCConfig
    label_encoder: ERCLabelEncoder

    def __init__(self, config: ERCConfig, label_encoder: ERCLabelEncoder) -> None:
        """
        Initialise the ERCDataCollator class with the given ERCLabelEncoder object.

        :param config: ERCConfig object containing the configuration
        :param label_encoder: ERCLabelEncoder object containing the label encoder
        """
        self.config = config
        self.label_encoder = label_encoder

    def __call__(self, record: List[MeldRecord]) -> dict:
        """
        Collates the data from the ERC dataset into a format that can be used and
        batched by the model, with multiple records turned into lists of its underlying
        datapoints. We also encode the labels to make it easier to use in the model.

        :param record: The list of records to collate
        :return: The collated data
        """
        x_text = [r.to_dialogue_prompt() for r in record]
        x_visual = self._visual_to_stacked_tensor(record)
        x_audio = self._audio_to_stacked_tensor(record)
        y_label = self.label_encoder([r.label for r in record])
        assert x_visual.shape == (len(record), self.config.visual_in_features)
        assert x_audio.shape == (len(record), self.config.audio_in_features)
        return {
            "x_text": x_text,
            "x_visual": x_visual,
            "x_audio": x_audio,
            ERCDataCollator.LABEL_NAME: y_label,
        }

    def _visual_to_stacked_tensor(self, record):
        """
        Stack together tensors representing feature vectors of images.

        :param record: The list of records to collate
        :return: The collated visual data as a tensor of shape (batch_size, visual_in_features)
        """
        return torch.stack([r.visual for r in record], dim=0)

    def _audio_to_stacked_tensor(self, record) -> torch.Tensor:
        """
        Stack together tensors of different sizes by truncating or padding them
        to the `target_size`.

        :param record: The list of records to collate
        :return: The collated audio data as a tensor of shape (batch_size, audio_in_features)
        """

        def truncate_or_pad(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
            # If the tensor is larger than the target size, truncate it
            if tensor.size(0) > target_size:
                return tensor[:target_size]
            # If the tensor is smaller, pad it
            elif tensor.size(0) < target_size:
                padding_size = target_size - tensor.size(0)
                padding = torch.zeros(padding_size, *tensor.size()[1:], dtype=tensor.dtype)
                return torch.cat([tensor, padding], dim=0)
            else:
                return tensor

        vecs = [torch.rand(10, 5), torch.rand(12, 5), torch.rand(8, 5)]
        vecs = [truncate_or_pad(tensor, target_size=self.config.audio_in_features) for tensor in vecs]
        return torch.stack(vecs, dim=0)
