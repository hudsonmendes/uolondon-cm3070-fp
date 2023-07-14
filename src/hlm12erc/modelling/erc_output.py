# Third-Party Libraries
from transformers.modeling_outputs import ModelOutput


class ERCOutput(ModelOutput):
    """
    ERCOutput is a class containing the outputs of the ERC model.
    """

    def __init__(
        self,
        loss=None,
        logits=None,
        hidden_states=None,
        attentions=None,
    ):
        """
        Constructor for ERCOutput class.

        :param loss: Optional loss value
        :param logits: Optional logits
        :param hidden_states: Optional hidden states
        :param attentions: Optional attentions
        """
        super().__init__(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )
