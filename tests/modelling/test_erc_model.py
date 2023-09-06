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
from hlm12erc.training.meld_record_preprocessor_audio import MeldAudioPreprocessorToWaveform
from hlm12erc.training.meld_record_preprocessor_visual import MeldVisualPreprocessorFilepathToResnet50


class TestERCModel(unittest.TestCase):
    def setUp(self):
        self.ff_layers = [ERCConfigFeedForwardLayer(out_features=14, dropout=0.5)]
        self.classes = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
        self.config = ERCConfig(feedforward_layers=self.ff_layers, classifier_classes=self.classes)

        self.label_encoder = ERCLabelEncoder(classes=self.classes)
        self.data_collator = ERCDataCollator(config=self.config, label_encoder=self.label_encoder)
        self.model = ERCModel(self.config, label_encoder=self.label_encoder)

        self.preprocessor_visual = MeldVisualPreprocessorFilepathToResnet50()
        self.preprocessor_audio = MeldAudioPreprocessorToWaveform()
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

    def test_forward_shape_labels(self):
        out = self.model.forward(**self.x)
        self.assertEqual(out.labels.shape, (len(self.x["x_text"]), len(self.classes)))

    def test_forward_shape_fusion_concat_with_only_text(self):
        config = dict(modules_fusion="concat", modules_visual_encoder="none", modules_audio_encoder="none")
        self._test_forward_shape_with_custom_config(config)

    def test_forward_shape_fusion_concat_with_only_visual(self):
        config = dict(modules_fusion="concat", modules_text_encoder="none", modules_audio_encoder="none")
        self._test_forward_shape_with_custom_config(config)

    def test_forward_shape_fusion_concat_with_only_audio(self):
        config = dict(modules_fusion="concat", modules_text_encoder="none", modules_visual_encoder="none")
        self._test_forward_shape_with_custom_config(config)

    def test_forward_shape_fusion_concat_with_text_and_visual(self):
        config = dict(modules_fusion="concat", modules_audio_encoder="none")
        self._test_forward_shape_with_custom_config(config)

    def test_forward_shape_fusion_concat_with_text_and_audio(self):
        config = dict(modules_fusion="concat", modules_visual_encoder="none")
        self._test_forward_shape_with_custom_config(config)

    def test_forward_shape_fusion_concat_with_audio_and_audio(self):
        config = dict(modules_fusion="concat", modules_text_encoder="none")
        self._test_forward_shape_with_custom_config(config)

    def test_forward_shape_fusion_mha_with_only_text(self):
        self._test_forward_shape_with_custom_config(
            dict(
                modules_fusion="multi_headed_attn",
                fusion_attention_heads_degree=2,
                fusion_out_features=512,
                modules_visual_encoder="none",
                modules_audio_encoder="none",
            )
        )

    def test_forward_shape_fusion_mha_with_only_visual(self):
        self._test_forward_shape_with_custom_config(
            dict(
                modules_fusion="multi_headed_attn",
                fusion_attention_heads_degree=2,
                fusion_out_features=512,
                modules_text_encoder="none",
                modules_audio_encoder="none",
            )
        )

    def test_forward_shape_fusion_mha_with_only_audio(self):
        self._test_forward_shape_with_custom_config(
            dict(
                modules_fusion="multi_headed_attn",
                fusion_attention_heads_degree=2,
                fusion_out_features=512,
                modules_text_encoder="none",
                modules_visual_encoder="none",
            )
        )

    def test_forward_shape_fusion_mha_with_text_and_visual(self):
        self._test_forward_shape_with_custom_config(
            dict(
                modules_fusion="multi_headed_attn",
                fusion_attention_heads_degree=2,
                fusion_out_features=512,
                modules_audio_encoder="none",
            )
        )

    def test_forward_shape_fusion_mha_with_text_and_audio(self):
        self._test_forward_shape_with_custom_config(
            dict(
                modules_fusion="multi_headed_attn",
                fusion_attention_heads_degree=2,
                fusion_out_features=512,
                modules_visual_encoder="none",
            )
        )

    def test_forward_shape_fusion_mha_with_audio_and_audio(self):
        self._test_forward_shape_with_custom_config(
            dict(
                modules_fusion="multi_headed_attn",
                fusion_attention_heads_degree=2,
                fusion_out_features=512,
                modules_text_encoder="none",
            )
        )

    def test_forward_shape_output_logits_shape(self):
        out = self.model.forward(**self.x)
        self.assertEqual(out.logits.shape, (len(self.x["x_text"]), len(self.classes)))

    def test_forward_shape_output_hidden_state_shape(self):
        out = self.model.forward(**self.x)
        self.assertEqual(
            out.hidden_states.shape,
            (len(self.x["x_text"]), self.config.feedforward_layers[-1].out_features),
        )

    def test_forward_shape_output_type_dict(self):
        out = self.model.forward(return_dict=True, **self.x)
        self.assertIsInstance(out, ERCOutput)

    def test_forward_shape_output_type_tuple(self):
        out = self.model.forward(return_dict=False, **self.x)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 5)

    def test_forward_shape_and_backward(self):
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

    def test_forward_output_hidden_state_l2_norm_None(self):
        norms = torch.norm(self._test_forward_with_custom_config(dict(feedforward_l2norm=None)).hidden_states, dim=1)
        [self.assertNotAlmostEqual(norm.item(), 1.0, places=5) for norm in norms]

    def test_forward_output_hidden_state_l2_norm_False(self):
        norms = torch.norm(self._test_forward_with_custom_config(dict(feedforward_l2norm=False)).hidden_states, dim=1)
        [self.assertNotAlmostEqual(norm.item(), 1.0, places=5) for norm in norms]

    def test_forward_output_hidden_state_l2_norm_True(self):
        norms = torch.norm(self._test_forward_with_custom_config(dict(feedforward_l2norm=True)).hidden_states, dim=1)
        [self.assertAlmostEqual(norm.item(), 1.0, places=5) for norm in norms]

    def _test_forward_shape_with_custom_config(self, updating_config: dict):
        out = self._test_forward_with_custom_config(updating_config)
        self.assertEqual(out.labels.shape, (len(self.x["x_text"]), len(self.classes)))

    def _test_forward_with_custom_config(self, updating_config: dict) -> ERCOutput:
        custom_config = self.config.to_dict()
        custom_config.update(updating_config)
        custom_model = ERCModel(ERCConfig(**custom_config), label_encoder=self.label_encoder)
        return custom_model.forward(**self.x)
