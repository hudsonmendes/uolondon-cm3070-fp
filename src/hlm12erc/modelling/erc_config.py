# Python Built-in Modules
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class ERCTextEmbeddingType:
    """Enumerates all available text embedding types for the model."""

    NONE = "none"
    GLOVE = "glove"
    GPT2 = "gpt2"


class ERCVisualEmbeddingType:
    """Enumerates all available visual embedding types for the model."""

    NONE = "none"
    RESNET50 = "resnet50"


class ERCAudioEmbeddingType:
    """Enumerates all available audio embedding types for the model."""

    NONE = "none"
    WAVEFORM = "waveform"
    WAV2VEC2 = "wav2vec2"


class ERCFusionTechnique:
    """Enumerates all available fusion techniques for the model."""

    CONCATENATION = "concat"


class ERCLossFunctions:
    """Enumerates all available loss functions for the model."""

    CATEGORICAL_CROSS_ENTROPY = "cce"
    DICE_COEFFICIENT = "dice"
    FOCAL_MULTI_CLASS_LOG = "focal"


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
    classifier_name: str = "untagged"
    classifier_learning_rate: float = 5e-5
    classifier_weight_decay: float = 0.1
    classifier_warmup_steps: int = 500
    classifier_epsilon: float = 1e-8
    classifier_loss_fn: str = ERCLossFunctions.CATEGORICAL_CROSS_ENTROPY

    losses_focal_alpha: List[float] | None = None
    losses_focal_gamma: float | None = None
    losses_focal_reduction: str | None = None

    modules_text_encoder: str = ERCTextEmbeddingType.GLOVE
    modules_visual_encoder: str = ERCVisualEmbeddingType.RESNET50
    modules_audio_encoder: str = ERCAudioEmbeddingType.WAVEFORM
    modules_fusion: str = ERCFusionTechnique.CONCATENATION

    text_in_features: int = 50  # 300 is the largest model
    text_out_features: int = 50  # must match in_features for GloVe
    text_limit_to_n_last_tokens: int | None = None  # truncation to most recent dialogue

    audio_in_features: int = 100_000  # 300_000 fits the all audio files
    audio_out_features: int = 512

    visual_in_features: Tuple[int, ...] = (3, 256, 721)  # required by resnet
    visual_out_features: int = -1  # defined by resnet50, get it from the embedding class

    feedforward_layers: Optional[List["ERCConfigFeedForwardLayer"]] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class ERCConfigFeedForwardLayer:
    """
    Represents a layer in the feedforward network
    """

    out_features: Optional[int] = None
    dropout: Optional[float] = None
