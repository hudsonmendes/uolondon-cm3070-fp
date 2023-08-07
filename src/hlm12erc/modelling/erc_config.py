# Python Built-in Modules
from dataclasses import dataclass
from typing import List, Optional


class ERCTextEmbeddingType:
    """Enumerates all available text embedding types for the model."""

    GLOVE = "glove"


class ERCVisualEmbeddingType:
    """Enumerates all available visual embedding types for the model."""

    RESNET50 = "resnet50"


class ERCAudioEmbeddingType:
    """Enumerates all available audio embedding types for the model."""

    WAVEFORM = "waveform"


class ERCFusionTechnique:
    """Enumerates all available fusion techniques for the model."""

    CONCATENATION = "concat"


class ERCLossFunctions:
    """Enumerates all available loss functions for the model."""

    CATEGORICAL_CROSS_ENTROPY = "cce"


@dataclass(frozen=True)
class ERCConfig:
    """
    ERCConfig is a class that defines the configuration for ERC model.
    It is used to instantiate an ERC model according to the specified arguments,
    defining the model architecture.

    The classifier's objective is collapse highly-dimensional representations
    found in the audio-visual + textual raw data into a probability distribution
    ranging across the different emotion labels. These emotions are
    - anger,
    - disgust,
    - fear,
    - joy,
    - neutral
    - sadness
    - surprise
    """

    classifier_classes: List[str]
    classifier_loss_fn: str = ERCLossFunctions.CATEGORICAL_CROSS_ENTROPY

    modules_text_encoder: str = ERCTextEmbeddingType.GLOVE
    modules_visual_encoder: str = ERCVisualEmbeddingType.RESNET50
    modules_audio_encoder: str = ERCAudioEmbeddingType.WAVEFORM
    modules_fusion: str = ERCFusionTechnique.CONCATENATION

    text_in_features: int = 50  # 300 is the largest model
    text_out_features: int = 50  # must match in_features for GloVe

    audio_in_features: int = 100_000  # 300_000 fits the all audio files
    audio_out_features: int = 512

    visual_in_features: int = -1  # defined by resnet50
    visual_out_features: int = -1  # defined by resnet50

    feedforward_layers: Optional[List["ERCConfigFeedForwardLayer"]] = None


@dataclass(frozen=True)
class ERCConfigFeedForwardLayer:
    """
    Represents a layer in the feedforward network
    """

    out_features: Optional[int] = None
    dropout: Optional[float] = None
