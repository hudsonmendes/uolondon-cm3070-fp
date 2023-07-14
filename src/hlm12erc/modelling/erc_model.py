# Python Built-in Modules
from typing import Optional, Tuple, Union

# Third-Party Libraries
import torch

# Local Folders
from .erc_config import ERCConfig
from .erc_emb_audio import ERCAudioEmbeddings
from .erc_emb_text import ERCTextEmbeddings
from .erc_emb_visual import ERCVisualEmbeddings
from .erc_fusion import ERCFusion
from .erc_output import ERCOutput


class ERCModel(torch.Module):
    """
    PyTorch implementation of a Multi-modal Modal capable of
    Emotion Recognition in Converations (or "ERC")
    """

    def __init__(self, config: ERCConfig) -> None:
        super().__init__()
        self.text_embeddings = ERCTextEmbeddings.resolve_type_from(config.embeds_text_encoder)(config)
        self.visual_embeddings = ERCVisualEmbeddings.resolve_type_from(config.embeds_visual_encoder)(config)
        self.audio_embeddings = ERCAudioEmbeddings.resolve_type_from(config.embeds_audio_encoder)(config)
        self.fusion_network = ERCFusion()
        self.feedforward = ERCModel._create_feedforward_based_on(config)

    @staticmethod
    def _create_feedforward_based_on(config: ERCConfig) -> torch.nn.Module:
        """
        Creates a feedforward network based on the given ERCConfig.
        """
        raise NotImplementedError("Not yet implemented")

    def forward(
        self,
        x_text: torch.Tensor,
        x_visual: torch.Tensor,
        x_audio: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[ERCOutput, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Main ERC model pipeline that runs the input through
        the different Feature Extraction modules and combines
        them before transforming and collapsing representations
        into the output softmax probability distribution of
        the different emotion labels.
        """
        raise NotImplementedError("Not yet implemented")
