# Python Built-in Modules
from typing import List

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling import (
    ERCAudioEmbeddingType,
    ERCConfig,
    ERCLabelEncoder,
    ERCTextEmbeddingType,
    ERCVisualEmbeddingType,
)

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
    device: torch.device | None

    def __init__(self, config: ERCConfig, label_encoder: ERCLabelEncoder) -> None:
        """
        Initialise the ERCDataCollator class with the given ERCLabelEncoder object.

        :param config: ERCConfig object containing the configuration
        :param label_encoder: ERCLabelEncoder object containing the label encoder
        """
        self.config = config
        self.label_encoder = label_encoder

    def __call__(
        self,
        batch: List[MeldRecord],
        device: torch.device | None = None,
    ) -> dict:
        """
        Collates the data from the ERC dataset into a format that can be used and
        batched by the model, with multiple records turned into lists of its underlying
        datapoints. We also encode the labels to make it easier to use in the model.

        :param record: The list of records to collate
        :param device: if provided, send tensors to device
        :return: The collated data
        """
        y_label = self.label_encoder([r.label for r in batch])

        x_text = None
        if self.config.modules_text_encoder != ERCTextEmbeddingType.NONE:
            x_text = [r.text for r in batch]

        x_visual = None
        if self.config.modules_visual_encoder != ERCVisualEmbeddingType.NONE:
            x_visual = self._visual_to_stacked_tensor([r.visual for r in batch])

        x_audio = None
        if self.config.modules_audio_encoder != ERCAudioEmbeddingType.NONE:
            x_audio = self._audio_to_stacked_tensor([r.audio for r in batch])

        if device is not None:
            y_label = y_label.to(device)
            if x_visual is not None:
                x_visual = x_visual.to(device)
            if x_audio is not None:
                x_audio = x_audio.to(device)
        return {
            "x_text": x_text,
            "x_visual": x_visual,
            "x_audio": x_audio,
            ERCDataCollator.LABEL_NAME: y_label,
        }

    def _visual_to_stacked_tensor(self, videos: List[torch.Tensor]) -> torch.Tensor:
        """
        Stack together tensors representing feature vectors of images.

        :param record: The list of individual tensors for visual data
        :return: The collated visual data as a tensor of shape (batch_size, *visual_in_features)
        """
        return torch.stack(videos, dim=0)

    def _audio_to_stacked_tensor(self, audios: List[torch.from_numpy]) -> torch.Tensor:
        """
        Stack together tensors of different sizes by truncating or padding them
        to the `target_size`.

        :param record: The list of individual tensors for audio data
        :return: The collated audio data as a tensor of shape (batch_size, audio_in_features)
        """

        def truncate_or_pad(vec: torch.Tensor, target_size: int) -> torch.Tensor:
            if vec.size(0) > target_size:
                return vec[:target_size]
            elif vec.size(0) < target_size:
                pad_len = target_size - vec.size(0)
                padding = torch.zeros(pad_len, *vec.size()[1:], dtype=vec.dtype)
                return torch.cat([vec, padding], dim=0)
            else:
                return vec

        vecs = [truncate_or_pad(audio, target_size=self.config.audio_in_features) for audio in audios]
        return torch.stack(vecs, dim=0)
