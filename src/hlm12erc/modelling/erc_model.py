# Python Built-in Modules
from typing import Optional, Tuple, Union

# Third-Party Libraries
import torch

# Local Folders
from .erc_config import ERCConfig
from .erc_emb_audio import ERCAudioEmbeddings
from .erc_emb_text import ERCTextEmbeddings
from .erc_emb_visual import ERCVisualEmbeddings
from .erc_feedforward import ERCFeedForwardConfig, ERCFeedForwardModel
from .erc_fusion import ERCFusion
from .erc_output import ERCOutput


class ERCModel(torch.nn.Module):
    """
    PyTorch implementation of a Multi-modal Modal capable of
    Emotion Recognition in Converations (or "ERC")
    """

    def __init__(self, config: ERCConfig) -> None:
        super().__init__()
        # Embedding Modules
        self.text_embeddings = ERCTextEmbeddings.resolve_type_from(config.modules_text_encoder)(config)
        self.visual_embeddings = ERCVisualEmbeddings.resolve_type_from(config.modules_visual_encoder)(config)
        self.audio_embeddings = ERCAudioEmbeddings.resolve_type_from(config.modules_audio_encoder)(config)
        # Fusion Network
        self.fusion_network = ERCFusion.resolve_type_from(config.modules_fusion)(
            embeddings=[self.text_embeddings, self.visual_embeddings, self.audio_embeddings],
            config=config,
        )
        # Feed Forward Transformation
        self.feedforward = ERCFeedForwardModel(
            in_features=self.fusion_network.out_features,
            config=ERCFeedForwardConfig(
                hidden_size=config.feedforward_hidden_size,
                num_layers=config.feedforward_num_layers,
                dropout=config.feedforward_dropout,
                activation=config.feedforward_activation,
            ),
        )
        # Softmax Activation
        self.softmax = torch.nn.Softmax(config.classifier_n_classes)

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

        y_text = self.text_embeddings(x_text)
        y_visual = self.visual_embeddings(x_visual)
        y_audio = self.audio_embeddings(x_audio)
        y_fusion = self.fusion_network(y_text, y_visual, y_audio)
        y_transformed = self.feedforward(y_fusion)
        y_pred = self.softmax(y_transformed)
        return y_pred
