# Python Built-in Modules
from abc import ABC, abstractmethod, abstractproperty

# Third-Party Libraries
import torch

# Local Folders
from .erc_config import ERCConfig


class ERCEmbeddings(ABC, torch.nn.Module):
    """
    Defines the base contract for embedding modules that will be implemented
    and used for the purpose of Emotion Recognition in Conversations.

    Example:
        >>> from abc import ABC
        >>> from hlm12erc.modelling.erc_emb import ERCEmbeddings
        >>> class ERCAudioEmbeddings(ABC, ERCEmbeddings):
        >>>     pass
    """

    def __init__(self, config: ERCConfig, *args, **kwargs) -> None:
        """
        Contract for the constructor of classes implementing
        ERCEmbeddings, requiring the `config` to be passed in,
        but it is not stored, because it is not required by all
        implementations to keep record of the original config.

        :param config: the configuration of the model
        """
        super().__init__(*args, **kwargs)
        assert config is not None

    @abstractproperty
    def out_features(self) -> int:
        """
        When implemented, provides the output features of the embeddings,
        which is required by the next layer to size its inputs

        :returns: an integer representing the number of output dims.
        """
        raise NotImplementedError("Abstract property not implemtented in base class.")