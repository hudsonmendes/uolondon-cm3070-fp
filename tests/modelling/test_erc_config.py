# Python Built-in Modules
import unittest

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCAudioEmbeddingType, ERCConfig, ERCTextEmbeddingType, ERCVisualEmbeddingType


class TestERCConfig(unittest.TestCase):
    def test_is_text_modality_enabled(self):
        self.assertTrue(
            ERCConfig(
                classifier_classes=["a"],
                modules_text_encoder=ERCTextEmbeddingType.GLOVE,
            ).is_text_modality_enabled()
        )
        self.assertTrue(
            ERCConfig(
                classifier_classes=["a"],
                modules_text_encoder=ERCTextEmbeddingType.GPT2,
            ).is_text_modality_enabled()
        )

    def test_is_text_modality_disabled(self):
        self.assertFalse(
            ERCConfig(
                classifier_classes=["a"],
                modules_text_encoder=ERCTextEmbeddingType.NONE,
            ).is_text_modality_enabled()
        )
        self.assertFalse(
            ERCConfig(
                classifier_classes=["a"],
                modules_text_encoder=None,
            ).is_text_modality_enabled()
        )

    def test_is_visual_modality_enabled(self):
        self.assertTrue(
            ERCConfig(
                classifier_classes=["a"],
                modules_visual_encoder=ERCVisualEmbeddingType.RESNET50,
            ).is_visual_modality_enabled()
        )

    def test_is_visual_modality_disabled(self):
        self.assertFalse(
            ERCConfig(
                classifier_classes=["a"],
                modules_visual_encoder=ERCVisualEmbeddingType.NONE,
            ).is_visual_modality_enabled()
        )
        self.assertFalse(
            ERCConfig(
                classifier_classes=["a"],
                modules_visual_encoder=None,
            ).is_visual_modality_enabled()
        )

    def test_is_audio_modality_enabled(self):
        self.assertTrue(
            ERCConfig(
                classifier_classes=["a"],
                modules_audio_encoder=ERCAudioEmbeddingType.WAVEFORM,
            ).is_audio_modality_enabled()
        )
        self.assertTrue(
            ERCConfig(
                classifier_classes=["a"],
                modules_audio_encoder=ERCAudioEmbeddingType.WAV2VEC2,
            ).is_audio_modality_enabled()
        )

    def test_is_audio_modality_disabled(self):
        self.assertFalse(
            ERCConfig(
                classifier_classes=["a"],
                modules_audio_encoder=ERCAudioEmbeddingType.NONE,
            ).is_audio_modality_enabled()
        )
        self.assertFalse(
            ERCConfig(
                classifier_classes=["a"],
                modules_audio_encoder=None,
            ).is_audio_modality_enabled()
        )
