# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling import ERCConfig, ERCLabelEncoder
from hlm12erc.training.erc_data_collator import ERCDataCollator
from hlm12erc.training.meld_record import MeldRecord


class TestERCDataCollator(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(classifier_classes=["a", "b", "c"])

        self.record1 = MeldRecord(
            text="Hello",
            visual=torch.randn(*self.config.visual_in_features),
            audio=torch.randn(1000),
            label="a",
        )
        self.record2 = MeldRecord(
            text="Hi",
            visual=torch.randn(*self.config.visual_in_features),
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
