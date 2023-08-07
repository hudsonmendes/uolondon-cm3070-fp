# Python Built-in Modules
from abc import abstractmethod
from typing import List, Type

# Third-Party Libraries
import torch
import torchtext
from torch.nn.functional import normalize as l2_norm
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer

# Local Folders
from .erc_config import ERCConfig, ERCTextEmbeddingType
from .erc_emb import ERCEmbeddings


class ERCTextEmbeddings(ERCEmbeddings):
    """
    ERCTextEmbeddings is an abstract class that defines the interface for text
    embedding layers.

    Examples:
        >>> from hlm12erc.modelling.erc_emb_text import ERCTextEmbeddings
        >>> class ERCMyCustomTextEmbeddings(ERCTextEmbeddings):
    """

    @abstractmethod
    def forward(self, x: List[str]) -> torch.Tensor:
        """
        When implemented, this method should receive a list of text and
        return a matrix of tensors (batch_size, out_features).
        """
        raise NotImplementedError("The method 'forward' must be implemented.")

    @staticmethod
    def resolve_type_from(expression: str) -> Type["ERCTextEmbeddings"]:
        if expression == ERCTextEmbeddingType.GLOVE:
            return ERCGloveTextEmbeddings
        raise ValueError(f"The text embeddings '{expression}' is not supported.")


class ERCGloveTextEmbeddings(ERCTextEmbeddings):
    """
    ERCGloveTextEmbeddings is a class that implements the text embedding layer using GloVe embeddings
    and then the mean of the embeddings for each token in the text.

    Example:
        >>> from hlm12erc.modelling import ERCConfig, ERCTextEmbeddingType
        >>> from hlm12erc.modelling.erc_emb_text import ERCGloveTextEmbeddings
        >>> config = ERCConfig()
        >>> ERCGloveTextEmbeddings.resolve_type_from(ERCTextEmbeddingType.GLOVE)(config)
    """

    hidden_size: int

    def __init__(self, config: ERCConfig) -> None:
        """
        Initializes the ERCGloveTextEmbeddings class.

        :param config: The configuration for the ERC model.
        """
        super().__init__(config=config)
        self.hidden_size = config.text_out_features
        self.tokenizer = get_tokenizer("basic_english", language="en")
        self.glove = torchtext.vocab.GloVe(name="6B", dim=config.text_out_features)

    def forward(self, x: List[str]) -> torch.Tensor:
        """
        Performs a forward pass through the text embedding layer.

        :param x: The input tensor of shape (batch_size,).
        :return: The output tensor of shape (batch_size, hidden_size).
        """
        t = [self.tokenizer(text) for text in x]
        v = [[self.glove.get_vecs_by_tokens(t, lower_case_backup=True) for t in seq] for seq in t]
        v = [[vii for vii in vi if torch.any(vii != 0)] for vi in v]
        y = pad_sequence([torch.stack(seq) for seq in v], batch_first=True)
        y = torch.mean(y, dim=1)
        y = l2_norm(y, p=2, dim=1)
        return y

    @property
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the embedding transformation.
        """
        return self.hidden_size
