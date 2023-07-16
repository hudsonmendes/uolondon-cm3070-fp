# Python Built-in Modules
from typing import List, Optional, Tuple, Union
from wave import Wave_read as Wave

# Third-Party Libraries
import torch
from PIL.Image import Image

# Local Folders
from .erc_config import ERCConfig
from .erc_emb_audio import ERCAudioEmbeddings
from .erc_emb_text import ERCTextEmbeddings
from .erc_emb_visual import ERCVisualEmbeddings
from .erc_feedforward import ERCFeedForward
from .erc_fusion import ERCFusion
from .erc_loss import ERCLoss
from .erc_output import ERCOutput


class ERCModel(torch.nn.Module):
    """
    PyTorch implementation of a Multi-modal Modal capable of
    Emotion Recognition in Converations (or "ERC")
    """

    def __init__(self, config: ERCConfig) -> None:
        """
        Constructs the ERC model by initializing the different modules based on hyperparameter
        configuration for the text, visual and audio encoders, as well as for the fusion network,
        the shape of the feedforward network and the loss function.

        :param config: ERCConfig object containing the hyperparameters for the ERC model
        """
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
        self.feedforward = ERCFeedForward(
            in_features=self.fusion_network.out_features,
            out_features=config.feedforward_out_features,
            layers=config.feedforward_layers,
        )
        # Softmax Activation
        self.logits = torch.nn.Linear(
            in_features=config.feedforward_out_features,
            out_features=config.classifier_n_classes,
        )
        self.softmax = torch.nn.Softmax(dim=1)
        # Loss Function
        self.loss = ERCLoss.resolve_type_from(config.classifier_loss_fn)()

    def forward(
        self,
        x_text: List[str],
        x_visual: List[Image],
        x_audio: List[Wave],
        y_true: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[ERCOutput, tuple]:
        """
        Main ERC model pipeline that runs the input through
        the different Feature Extraction modules and combines
        them before transforming and collapsing representations
        into the output softmax probability distribution of
        the different emotion labels.

        :param x_text: List of strings containing the text input
        :param x_visual: List of PIL.Image objects containing the visual input
        :param x_audio: List of wave.Wave_read objects containing the audio input
        :param y_true: Optional tensor containing the true labels
        :param return_dict: Whether to return the output as a dictionary or tuple
        :return: ERCOutput object containing the loss, predicted labels, logits, hidden states and attentions
        """

        y_text = self.text_embeddings(x_text)
        y_visual = self.visual_embeddings(x_visual)
        y_audio = self.audio_embeddings(x_audio)
        y_fusion = self.fusion_network(y_text, y_visual, y_audio)
        y_transformed = self.feedforward(y_fusion)
        y_logits = self.logits(y_transformed)
        y_pred = self.softmax(y_logits)
        loss = self.loss(y_pred, y_true) if y_true is not None else None
        output = ERCOutput(loss=loss, labels=y_pred, logits=y_logits, hidden_states=y_transformed, attentions=None)
        return output if return_dict else output.to_tuple()
