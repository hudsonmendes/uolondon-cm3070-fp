# Python Built-in Modules
import unittest
import wave

# Third-Party Libraries
import torch
from PIL import Image

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig
from hlm12erc.modelling.erc_model import ERCModel
from hlm12erc.modelling.erc_output import ERCOutput


class TestERCModel(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig()
        self.model = ERCModel(self.config)
        self.y_true = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        self.x_text = ["that was a good one for a second there I was wow"]
        self.x_visual = [Image.open("tests/fixtures/d-1038-seq-17.png")]
        self.x_audio = [wave.open("tests/fixtures/d-1038-seq-17.wav")]

    def tearDown(self):
        del self.model

    def test_forward_output_labels_shape(self):
        out = self.model.forward(self.x_text, self.x_visual, self.x_audio, self.y_true)
        self.assertEqual(out.labels.shape, (len(self.x_text), self.config.classifier_n_classes))

    def test_forward_output_logits_shape(self):
        out = self.model.forward(self.x_text, self.x_visual, self.x_audio, self.y_true)
        self.assertEqual(out.logits.shape, (len(self.x_text), self.config.classifier_n_classes))

    def test_forward_output_hidden_state_shape(self):
        out = self.model.forward(self.x_text, self.x_visual, self.x_audio, self.y_true)
        self.assertEqual(out.hidden_states.shape, (len(self.x_text), self.config.feedforward_out_features))

    def test_forward_output_type_dict(self):
        out = self.model.forward(self.x_text, self.x_visual, self.x_audio, self.y_true, return_dict=True)
        self.assertIsInstance(out, ERCOutput)

    def test_forward_output_type_tuple(self):
        out = self.model.forward(self.x_text, self.x_visual, self.x_audio, self.y_true, return_dict=False)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 4)

    def test_forward_and_backward(self):
        # Compute output and loss
        out = self.model.forward(self.x_text, self.x_visual, self.x_audio, self.y_true)
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
