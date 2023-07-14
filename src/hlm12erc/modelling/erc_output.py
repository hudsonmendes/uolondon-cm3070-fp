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
        erc_logits=None,
        erc_hidden_states=None,
        erc_attentions=None,
    ):
        """
        Constructor for ERCOutput class.

        :param loss: Optional loss value
        :param logits: Optional logits
        :param hidden_states: Optional hidden states
        :param attentions: Optional attentions
        :param erc_logits: Optional ERC logits
        :param erc_hidden_states: Optional ERC hidden states
        :param erc_attentions: Optional ERC attentions
        """
        super().__init__(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )
        self.erc_logits = erc_logits
        self.erc_hidden_states = erc_hidden_states
        self.erc_attentions = erc_attentions
