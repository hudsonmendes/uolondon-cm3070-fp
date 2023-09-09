# Python Built-in Modules
from dataclasses import dataclass, field
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
    MULTI_HEADED_ATTENTION = "multi_headed_attn"


class ERCLossFunctions:
    """Enumerates all available loss functions for the model."""

    TRIPLET = "triplet"
    CROSSENTROPY = "cce"
    CROSSENTROPY_PLUS_TRIPLET = "cce+triplet"
    DICE = "dice"
    DICE_PLUS_TRIPET = "dice+triplet"
    FOCAL = "focal"
    FOCAL_PLUS_TRIPLET = "focal+triplet"


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

    classifier_classes: List[str] = field()
    classifier_name: str = field(default="untagged")
    classifier_learning_rate: float = field(default=5e-5)
    classifier_weight_decay: float = field(default=0.1)
    classifier_warmup_steps: int = field(default=500)
    classifier_epsilon: float = field(default=1e-8)
    classifier_loss_fn: str = field(default=ERCLossFunctions.CROSSENTROPY)
    classifier_early_stopping_patience: int | None = field(default=None)
    classifier_metric_for_best_model: str = field(default="loss")

    losses_focal_alpha: List[float] | None = field(default=None)
    losses_focal_gamma: float | None = field(default=None)
    losses_focal_reduction: str | None = field(default=None)

    modules_text_encoder: str = field(default=ERCTextEmbeddingType.GLOVE)
    modules_visual_encoder: str = field(default=ERCVisualEmbeddingType.RESNET50)
    modules_audio_encoder: str = field(default=ERCAudioEmbeddingType.WAVEFORM)
    modules_fusion: str = field(default=ERCFusionTechnique.CONCATENATION)

    text_in_features: int = field(default=50)  # 300 is the largest model
    text_out_features: int = field(default=50)  # must match in_features for GloVe
    text_limit_to_n_last_tokens: int | None = field(default=None)  # truncation to most recent dialogue

    audio_in_features: int = field(default=100_000)  # 300_000 fits the all audio files
    audio_out_features: int = field(default=512)

    visual_preprocess_faceonly: bool | None = field(default=None)
    visual_preprocess_retinaface_weights_path: str | None = field(default=None)
    visual_in_features: Tuple[int, ...] = field(default=(3, 256, 721))  # required by resnet
    visual_out_features: int = field(default=-1)  # defined by resnet50, get it from the embedding class

    fusion_attention_heads_degree: int | None = field(default=None)
    fusion_out_features: int | None = field(default=None)  # none for concat, fit RAM for mha

    feedforward_l2norm: bool = field(default=False)
    feedforward_layers: Optional[List["ERCConfigFeedForwardLayer"]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class ERCConfigFeedForwardLayer:
    """
    Represents a layer in the feedforward network
    """

    out_features: Optional[int] = None
    dropout: Optional[float] = None
