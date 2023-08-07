# Python Built-in Modules
import unittest

# My Packages and Modules
from hlm12erc.modelling.erc_config import (
    ERCAudioEmbeddingType,
    ERCConfig,
    ERCConfigFeedForwardLayer,
    ERCFusionTechnique,
    ERCLossFunctions,
    ERCTextEmbeddingType,
    ERCVisualEmbeddingType,
)
from hlm12erc.training.erc_config_formatter import ERCConfigFormatter


class TestERCConfig(unittest.TestCase):
    def test_str_feedforward_layers(self):
        # Test that __str__ includes feedforward layers if present
        config_with_ffl = ERCConfig(
            modules_text_encoder=ERCTextEmbeddingType.GLOVE,
            modules_visual_encoder=ERCVisualEmbeddingType.RESNET50,
            modules_audio_encoder=ERCAudioEmbeddingType.WAVEFORM,
            modules_fusion=ERCFusionTechnique.CONCATENATION,
            text_in_features=300,
            text_out_features=300,
            audio_in_features=325458,
            audio_out_features=512,
            feedforward_layers=[
                ERCConfigFeedForwardLayer(out_features=512),
                ERCConfigFeedForwardLayer(out_features=256),
            ],
            classifier_loss_fn=ERCLossFunctions.CATEGORICAL_CROSS_ENTROPY,
        )
        self.assertEqual(
            ERCConfigFormatter(config_with_ffl).represent(),
            "hlm12erc-glove-resnet50-waveform-concat-t300x300-a325458x512-ffl512+256-cce",
        )

    def test_str_feedforward_out_features(self):
        # Test that __str__ includes feedforward out features if present
        config_with_ff_out = ERCConfig(
            modules_text_encoder=ERCTextEmbeddingType.GLOVE,
            modules_visual_encoder=ERCVisualEmbeddingType.RESNET50,
            modules_audio_encoder=ERCAudioEmbeddingType.WAVEFORM,
            modules_fusion=ERCFusionTechnique.CONCATENATION,
            text_in_features=300,
            text_out_features=300,
            audio_in_features=325458,
            audio_out_features=512,
            feedforward_layers=None,
            classifier_loss_fn=ERCLossFunctions.CATEGORICAL_CROSS_ENTROPY,
        )
        self.assertEqual(
            ERCConfigFormatter(config_with_ff_out).represent(),
            "hlm12erc-glove-resnet50-waveform-concat-t300x300-a325458x512-ffldefault-cce",
        )
