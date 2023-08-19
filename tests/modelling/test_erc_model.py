# Python Built-in Modules
import unittest
import wave

# Third-Party Libraries
import torch
from PIL import Image

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig, ERCConfigFeedForwardLayer
from hlm12erc.modelling.erc_label_encoder import ERCLabelEncoder
from hlm12erc.modelling.erc_model import ERCModel
from hlm12erc.modelling.erc_output import ERCOutput
from hlm12erc.training.erc_data_collator import ERCDataCollator
from hlm12erc.training.meld_record import MeldRecord
from hlm12erc.training.meld_record_preprocessor_audio import MeldAudioPreprocessor
from hlm12erc.training.meld_record_preprocessor_visual import MeldVisualPreprocessor


class TestERCModel(unittest.TestCase):
    def setUp(self):
        self.ff_layers = [ERCConfigFeedForwardLayer(out_features=7, dropout=0.5)]
        self.classes = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
        self.config = ERCConfig(feedforward_layers=self.ff_layers, classifier_classes=self.classes)

        self.label_encoder = ERCLabelEncoder(classes=self.classes)
        self.data_collator = ERCDataCollator(config=self.config, label_encoder=self.label_encoder)
        self.model = ERCModel(self.config, label_encoder=self.label_encoder)

        self.preprocessor_visual = MeldVisualPreprocessor()
        self.preprocessor_audio = MeldAudioPreprocessor()
        self.x = self.data_collator(
            [
                MeldRecord(
                    text="A said: ###that was a good one for a second there I was wow###",
                    audio=self.preprocessor_audio(wave.open("tests/fixtures/d-1038-seq-17.wav")),
                    visual=self.preprocessor_visual(Image.open("tests/fixtures/d-1038-seq-17.png")),
                    label="joy",
                )
            ]
        )

    def tearDown(self):
        del self.model

    def test_attr_config(self):
        self.assertEqual(self.model.config, self.config)

    def test_attr_label_encoder(self):
        self.assertEqual(self.model.label_encoder, self.label_encoder)

    def test_forward_output_labels_shape(self):
        out = self.model.forward(**self.x)
        self.assertEqual(out.labels.shape, (len(self.x["x_text"]), len(self.classes)))

    def test_forward_output_success_with_only_text(self):
        out = ERCModel(
            label_encoder=self.label_encoder,
            config=ERCConfig(
                feedforward_layers=self.ff_layers,
                classifier_classes=self.classes,
                modules_visual_encoder="none",
                modules_audio_encoder="none",
            ),
        ).forward(**self.x)
        self.assertEqual(out.labels.shape, (len(self.x["x_text"]), len(self.classes)))

    def test_forward_output_success_with_only_visual(self):
        out = ERCModel(
            label_encoder=self.label_encoder,
            config=ERCConfig(
                feedforward_layers=self.ff_layers,
                classifier_classes=self.classes,
                modules_text_encoder="none",
                modules_audio_encoder="none",
            ),
        ).forward(**self.x)
        self.assertEqual(out.labels.shape, (len(self.x["x_text"]), len(self.classes)))

    def test_forward_output_success_with_only_audio(self):
        out = ERCModel(
            label_encoder=self.label_encoder,
            config=ERCConfig(
                feedforward_layers=self.ff_layers,
                classifier_classes=self.classes,
                modules_visual_encoder="none",
                modules_text_encoder="none",
            ),
        ).forward(**self.x)
        self.assertEqual(out.labels.shape, (len(self.x["x_text"]), len(self.classes)))

    def test_forward_output_logits_shape(self):
        out = self.model.forward(**self.x)
        self.assertEqual(out.logits.shape, (len(self.x["x_text"]), len(self.classes)))

    def test_forward_output_hidden_state_shape(self):
        out = self.model.forward(**self.x)
        self.assertEqual(
            out.hidden_states.shape,
            (len(self.x["x_text"]), self.config.feedforward_layers[-1].out_features),
        )

    def test_forward_output_type_dict(self):
        out = self.model.forward(return_dict=True, **self.x)
        self.assertIsInstance(out, ERCOutput)

    def test_forward_output_type_tuple(self):
        out = self.model.forward(return_dict=False, **self.x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 5)

    def test_forward_and_backward(self):
        # Compute output and loss
        out = self.model.forward(**self.x)
        loss = out.loss

        # Compute gradients
        self.model.zero_grad()
        loss.backward()

        # Check that gradients are not None
        for param in self.model.parameters():
            self.assertIsNotNone(param.grad)

        # Check that gradients are finite
        for param in self.model.parameters():
            self.assertTrue(torch.isfinite(param.grad).all())
