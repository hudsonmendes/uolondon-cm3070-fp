# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCAudioEmbeddingType, ERCConfig, ERCTextEmbeddingType, ERCVisualEmbeddingType
from hlm12erc.modelling.erc_label_encoder import ERCLabelEncoder
from hlm12erc.training.erc_data_collator import ERCDataCollator
from hlm12erc.training.meld_record import MeldRecord


class TestERCDataCollator(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(classifier_classes=["a", "b", "c"])

        self.record1 = MeldRecord(
            text="Hello",
            visual=torch.randn((3, 256, 721)),
            audio=torch.randn(1000),
            label="a",
        )
        self.record2 = MeldRecord(
            text="Hi",
            visual=torch.randn((3, 256, 721)),
            audio=torch.randn(1000),
            label="c",
        )

        self.batch = [self.record1, self.record2]
        self.label_encoder = ERCLabelEncoder(classes=self.config.classifier_classes)
        self.collator = ERCDataCollator(self.config, self.label_encoder)

    def tearDown(self):
        pass

    def test_result_is_dict(self):
        self.assertIsInstance(self.collator(self.batch), dict)

    def test_result_contains_x_text(self):
        self.assertIn("x_text", self.collator(self.batch))

    def test_result_contains_x_visual(self):
        self.assertIn("x_visual", self.collator(self.batch))

    def test_result_contains_x_audio(self):
        self.assertIn("x_audio", self.collator(self.batch))

    def test_result_contains_label_name(self):
        self.assertIn(ERCDataCollator.LABEL_NAME, self.collator(self.batch))

    def test_x_text_has_correct_length(self):
        self.assertEqual(len(self.collator(self.batch)["x_text"]), len(self.batch))

    def test_x_visual_has_correct_shape(self):
        self.assertEqual(
            self.collator(self.batch)["x_visual"].shape,
            (len(self.batch), *self.config.visual_in_features),
        )

    def test_x_audio_has_correct_shape(self):
        self.assertEqual(
            self.collator(self.batch)["x_audio"].shape,
            (len(self.batch), self.config.audio_in_features),
        )

    def test_label_name_has_correct_shape(self):
        self.assertEqual(
            self.collator(self.batch)[ERCDataCollator.LABEL_NAME].shape,
            (len(self.batch), len(self.config.classifier_classes)),
        )

    def test_unimodal_config_generates_None_for_other_modalities(self):
        config = ERCConfig(
            classifier_classes=["a", "b", "c"],
            modules_text_encoder=ERCTextEmbeddingType.NONE,
            modules_visual_encoder=ERCVisualEmbeddingType.NONE,
            modules_audio_encoder=ERCAudioEmbeddingType.NONE,
        )
        self.batch = [self.record1, self.record2]
        self.label_encoder = ERCLabelEncoder(classes=self.config.classifier_classes)
        self.collator = ERCDataCollator(config, self.label_encoder)
        actual = self.collator(self.batch)
        self.assertIsNone(actual["x_text"])
        self.assertIsNone(actual["x_audio"])
        self.assertIsNone(actual["x_visual"])
        self.assertIsNotNone(actual["y_true"])
