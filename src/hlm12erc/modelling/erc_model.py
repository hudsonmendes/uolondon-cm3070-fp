# Python Built-in Modules
from typing import List, Optional, Tuple, Union

# Third-Party Libraries
import torch

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
            out_features=config.classifier_n_classes,
            layers=config.feedforward_layers,
        )
        # Softmax Activation
        self.softmax = torch.nn.Softmax(config.classifier_n_classes)
        # Loss Function
        self.loss = ERCLoss.resolve_type_from(config.classifier_loss_fn)()

    def forward(
        self,
        x_text: List[str],
        x_visual: torch.Tensor,
        x_audio: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[ERCOutput, tuple]:
        """
        Main ERC model pipeline that runs the input through
        the different Feature Extraction modules and combines
        them before transforming and collapsing representations
        into the output softmax probability distribution of
        the different emotion labels.
        """

        y_transformed, y_pred = self._calculate_y_pred(x_text, x_visual, x_audio)
        output = ERCOutput(
            loss=self._calculate_loss(y_true, y_pred),
            logits=y_pred,
            hidden_states=y_transformed,
            attentions=None,
        )
        return output.to_tuple() if not return_dict else output

    def _calculate_y_pred(
        self,
        x_text: List[str],
        x_visual: torch.Tensor,
        x_audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Goes through the transformation graph and produces
        the final output of the model.

        :param x_text: Input text tensor
        :param x_visual: Input visual tensor
        :param x_audio: Input audio tensor
        :return: Tuple of (y_transformed, y_pred)
        """
        y_text = self.text_embeddings(x_text)
        y_visual = self.visual_embeddings(x_visual)
        y_audio = self.audio_embeddings(x_audio)
        y_fusion = self.fusion_network(y_text, y_visual, y_audio)
        y_transformed = self.feedforward(y_fusion)
        y_pred = self.softmax(y_transformed)
        return y_transformed, y_pred

    def _calculate_loss(self, y_true, y_pred):
        """
        Calculates the loss given the true and predicted labels,
        if the true labels are provided.

        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Loss tensor
        """
        loss = None
        if y_true is not None:
            loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)
        return loss
