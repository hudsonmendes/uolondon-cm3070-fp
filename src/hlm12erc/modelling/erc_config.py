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

    Attributes:
        classifier_classes: The list of emotion classes to be predicted by the classifier.
        classifier_name: The name of the classifier.
        classifier_learning_rate: The learning rate of the classifier.
        classifier_weight_decay: The weight decay of the classifier.
        classifier_warmup_steps: The number of warmup steps of the classifier.
        classifier_epsilon: The epsilon of the classifier.
        classifier_loss_fn: The loss function of the classifier.
        classifier_early_stopping_patience: The early stopping patience of the classifier.
        classifier_metric_for_best_model: The metric for the best model of the classifier.

        losses_focal_alpha: The alpha of the focal loss.
        losses_focal_gamma: The gamma of the focal loss.
        losses_focal_reduction: The reduction of the focal loss.

        modules_text_encoder: The text encoder module.
        modules_visual_encoder: The visual encoder module.
        modules_audio_encoder: The audio encoder module.
        modules_fusion: The fusion module.

        text_in_features: The input features of the text encoder.
        text_out_features: The output features of the text encoder.
        text_limit_to_n_last_tokens: The number of last tokens to be considered in the text encoder.

        audio_in_features: The input features of the audio encoder, 300_000 fits all audio files.
        audio_out_features: The output features of the audio encoder.

        visual_preprocess_faceonly: Whether to preprocess the visual input to only include faces.
        visual_preprocess_retinaface_weights_path: The path to the retinaface weights.
        visual_in_features: The input features of the visual encoder, resnet requires (3, 256, 721)
        visual_out_features: The output features of the visual encoder.

        fusion_attention_heads_degree: The number of attention heads in the fusion module.
        fusion_out_features: The output features of the fusion module.

        feedforward_l2norm: Whether to apply l2 normalization to the feedforward network.
        feedforward_layers: The layers of the feedforward network.
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

    modules_text_encoder: str = field(default=ERCTextEmbeddingType.NONE)
    modules_visual_encoder: str = field(default=ERCVisualEmbeddingType.NONE)
    modules_audio_encoder: str = field(default=ERCAudioEmbeddingType.NONE)
    modules_fusion: str = field(default=ERCFusionTechnique.CONCATENATION)

    text_in_features: int | None = field(default=None)
    text_out_features: int | None = field(default=None)
    text_limit_to_n_last_tokens: int | None = field(default=None)

    audio_in_features: int | None = field(default=None)
    audio_out_features: int | None = field(default=None)

    visual_preprocess_faceonly: bool | None = field(default=None)
    visual_preprocess_retinaface_weights_path: str | None = field(default=None)
    visual_in_features: Tuple[int, ...] | None = field(default=None)
    visual_out_features: int | None = field(default=None)

    fusion_attention_heads_degree: int | None = field(default=None)
    fusion_out_features: int | None = field(default=None)

    feedforward_l2norm: bool = field(default=False)
    feedforward_layers: Optional[List["ERCConfigFeedForwardLayer"]] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def is_text_modality_enabled(self) -> bool:
        return self.modules_text_encoder is not None and self.modules_text_encoder != ERCTextEmbeddingType.NONE

    def is_visual_modality_enabled(self) -> bool:
        return self.modules_visual_encoder is not None and self.modules_visual_encoder != ERCVisualEmbeddingType.NONE

    def is_audio_modality_enabled(self) -> bool:
        return self.modules_audio_encoder is not None and self.modules_audio_encoder != ERCAudioEmbeddingType.NONE


@dataclass(frozen=True)
class ERCConfigFeedForwardLayer:
    """
    Represents a layer in the feedforward network
    """

    out_features: Optional[int] = None
    dropout: Optional[float] = None
