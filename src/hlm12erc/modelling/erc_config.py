# Local Folders
from .erc_emb_audio import ERCAudioEmbeddingType
from .erc_emb_text import ERCTextEmbeddingType
from .erc_emb_visual import ERCVisualEmbeddingType
from .erc_feedforward import ERCFeedForwardActivation


class ERCConfig:
    """
    ERCConfig is a class that defines the configuration for ERC model.
    It is used to instantiate an ERC model according to the specified arguments,
    defining the model architecture.
    """

    embeds_text_encoder: str = ERCTextEmbeddingType.GLOVE
    embeds_visual_encoder: str = ERCVisualEmbeddingType.RESNET_50
    embeds_audio_encoder: str = ERCAudioEmbeddingType.WAVEFORM

    text_hidden_size: int = 300

    audio_in_features: int = 1
    audio_hidden_size: int = 64
    audio_num_layers: int = 3
    audio_dropout: float = 0.1
    audio_activation: str = ERCFeedForwardActivation.RELU

    feedforward_hidden_size: int = 768
    feedforward_num_layers: int = 3
    feedforward_dropout: float = 0.1
    feedforward_activation: str = ERCFeedForwardActivation.RELU
