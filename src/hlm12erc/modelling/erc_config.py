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

    modules_text_encoder: str = ERCTextEmbeddingType.GLOVE
    modules_visual_encoder: str = ERCVisualEmbeddingType.RESNET50
    modules_audio_encoder: str = ERCAudioEmbeddingType.WAVEFORM
    modules_fusion: str = ERCFusionTechnique.CONCATENATION

    text_in_features: int = 300
    text_out_features: int = 300

    audio_in_features: int = 325458
    audio_out_features: int = 512

    feedforward_layers: Optional[List["ERCConfigFeedForwardLayer"]] = None
    feedforward_out_features: int = 1024

    classifier_n_classes: int = 7
    classifier_loss_fn: str = ERCLossFunctions.CATEGORICAL_CROSS_ENTROPY

    def __str__(self) -> str:
        ff_layerspec = (
            "+".join([str(layer.out_features) for layer in self.feedforward_layers])
            if self.feedforward_layers
            else "default"
        )
        return "-".join(
            [
                "hlm12erc",
                self.modules_text_encoder.lower(),
                self.modules_visual_encoder.lower(),
                self.modules_audio_encoder.lower(),
                self.modules_fusion.lower(),
                f"t{self.text_in_features}x{self.text_out_features}",
                f"a{self.audio_in_features}x{self.audio_out_features}",
                f"ff{self.feedforward_out_features}",
                f"ffl{ff_layerspec}",
                f"{self.classifier_loss_fn}",
            ]
        )

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class ERCConfigFeedForwardLayer:
    """
    Represents a layer in the feedforward network
    """

    out_features: Optional[int] = None
    dropout: Optional[float] = None
