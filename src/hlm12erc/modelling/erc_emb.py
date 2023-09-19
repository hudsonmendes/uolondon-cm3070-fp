# Python Built-in Modules
from abc import ABC, abstractproperty

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

    config: ERCConfig
    _device: torch.device | None

    def __init__(self, config: ERCConfig, *args, **kwargs) -> None:
        """
        Contract for the constructor of classes implementing
        ERCEmbeddings, requiring the `config` to be passed in,
        but it is not stored, because it is not required by all
        implementations to keep record of the original config.

        :param config: the configuration of the model
        """
        super(ERCEmbeddings, self).__init__(*args, **kwargs)
        self._device = None
        self.config = config
        assert config is not None

    def cache_or_get_same_device_as(self, module: torch.nn.Module) -> torch.device | None:
        """
        Unless self._device has a value `next(module.parameters()).device` and
        caches it as self._device, then returns self._device.

        :param module: the module to get the device from
        :return: the device of the module
        """
        if self._device is None:
            next_param = next(module.parameters(), None)
            if next_param is not None:
                self._device = next_param.device
        return self._device

    @abstractproperty
    def out_features(self) -> int:
        """
        Returns the dimensionality of the vectors produced by the embedding transformation.
        """
        raise NotImplementedError("The property 'out_features' must be implemented.")
