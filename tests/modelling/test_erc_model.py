# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig
from hlm12erc.modelling.erc_model import ERCModel


class TestERCModel(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(
            modules_text_encoder="my_project.erc_emb_text.ERCTextEmbeddings",
            modules_visual_encoder="my_project.erc_emb_visual.ERCVisualEmbeddings",
            modules_audio_encoder="my_project.erc_emb_audio.ERCAudioEmbeddings",
            modules_fusion="my_project.erc_fusion.ERCFusion",
            feedforward_hidden_size=128,
            feedforward_num_layers=2,
            feedforward_dropout=0.1,
            feedforward_activation="relu",
            classifier_n_classes=7,
        )
        self.model = ERCModel(self.config)

    def tearDown(self):
        del self.model

    def test_forward_output_shape(self):
        x_text = torch.randn((32, 100))
        x_visual = torch.randn((32, 200))
        x_audio = torch.randn((32, 300))
        out = self.model.forward(x_text, x_visual, x_audio)
        self.assertEqual(out.logits.y_pred.shape, (32, self.config.classifier_n_classes))

    def test_forward_output_range(self):
        x_text = torch.randn((32, 100))
        x_visual = torch.randn((32, 200))
        x_audio = torch.randn((32, 300))
        y_pred = self.model.forward(x_text, x_visual, x_audio)
        self.assertTrue(torch.all(y_pred >= 0))
        self.assertTrue(torch.all(y_pred <= 1))

    def test_forward_output_sum(self):
        x_text = torch.randn((32, 100))
        x_visual = torch.randn((32, 200))
        x_audio = torch.randn((32, 300))
        y_pred = self.model.forward(x_text, x_visual, x_audio)
        self.assertTrue(torch.allclose(y_pred.sum(dim=1), torch.ones((32,))))

    def test_forward_output_grad(self):
        x_text = torch.randn((32, 100), requires_grad=True)
        x_visual = torch.randn((32, 200), requires_grad=True)
        x_audio = torch.randn((32, 300), requires_grad=True)
        y_pred = self.model.forward(x_text, x_visual, x_audio)
        loss = y_pred.sum()
        loss.backward()
        self.assertIsNotNone(x_text.grad)
        self.assertIsNotNone(x_visual.grad)
        self.assertIsNotNone(x_audio.grad)
