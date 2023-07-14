# Python Built-in Modules
from abc import ABC
from typing import Type

# Third-Party Libraries
import torch
import torchtext
from nltk import word_tokenize

# Local Folders
from .erc_config import ERCConfig


class ERCTextEmbeddingType:
    """
    Enumerates the available implementations for the text embedding layer.
    """

    GLOVE = "glove"


class ERCTextEmbeddings(ABC, torch.nn.Module):
    """
    ERCTextEmbeddings is an abstract class that defines the interface for text embedding layers.

    Examples:
        >>> from hlm12erc.modelling.erc_emb_text import ERCTextEmbeddingType, ERCTextEmbeddings
        >>> ERCTextEmbeddings.resolve_type_from(ERCTextEmbeddingType.GLOVE)
    """

    config: ERCConfig

    def __init__(self, config: ERCConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

    @staticmethod
    def resolve_type_from(expression: str) -> Type["ERCTextEmbeddings"]:
        if expression == ERCTextEmbeddingType.GLOVE:
            return ERCGloveTextEmbeddings
        raise ValueError(f"Unknown text embedding type: {expression}")


class ERCGloveTextEmbeddings(ERCTextEmbeddings):
    """
    ERCGloveTextEmbeddings is a class that implements the text embedding layer using GloVe embeddings
    and then the mean of the embeddings for each token in the text.

    Examples:
        >>> from hlm12erc.modelling.erc_config import ERCConfig
        >>> from hlm12erc.modelling.erc_emb_text import ERCGloveTextEmbeddings
        >>> config = ERCConfig()
        >>> embeddings = ERCGloveTextEmbeddings(config)
        >>> embeddings(["This is a sentence.", "This is another sentence."])
    """

    def __init__(self, config: ERCConfig) -> None:
        """
        Initializes the ERCGloveTextEmbeddings class.

        :param config: The configuration for the ERC model.
        """
        super().__init__(config)
        self.glove = torchtext.vocab.GloVe(name="6B", dim=config.text_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the text embedding layer.

        :param x: The input tensor of shape (batch_size,).
        :return: The output tensor of shape (batch_size, hidden_size).
        """
        y = [word_tokenize(text) for text in x]
        y = [[self.glove.get_vecs_by_tokens(t) for t in seq] for seq in y]
        y = torch.tensor(y, dtype=torch.float)
        y = torch.mean(y, dim=2)
        return y
