# Python Built-in Modules
from typing import Optional

# Third-Party Libraries
import torch
from transformers.modeling_outputs import ModelOutput


class ERCOutput(ModelOutput):
    """
    ERCOutput is a class containing the outputs of the ERC model.
    """

    def __init__(
        self,
        labels: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attentions: Optional[torch.Tensor] = None,
    ):
        """
        Constructor for ERCOutput class.

        :param labels: Softmax Probability Distribution of Emotion labels
        :param loss: Loss calculated between the labels predicted and expected
        :param logits: The fused embeddings transformed by the feedforward network
        :param hidden_states: The fused embeddings before transformation
        :param attentions: Attention weights, if available for the fusion layer implementation
        """
        super().__init__(
            loss=loss,
            labels=labels,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions if attentions else (),
        )
