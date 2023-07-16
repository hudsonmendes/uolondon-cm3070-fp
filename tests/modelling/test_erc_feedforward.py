# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig, ERCConfigFeedForwardLayer
from hlm12erc.modelling.erc_feedforward import ERCFeedForward


class TestERCFeedForwardModel(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(
            feedforward_layers=[
                ERCConfigFeedForwardLayer(out_features=32, dropout=0.2),
                ERCConfigFeedForwardLayer(out_features=16, dropout=0.2),
                ERCConfigFeedForwardLayer(out_features=8, dropout=0.1),
                ERCConfigFeedForwardLayer(out_features=4),
            ]
        )
        self.model = ERCFeedForward(
            in_features=64,
            layers=self.config.feedforward_layers,
        )

    def tearDown(self):
        del self.model

    def test_forward_shape(self):
        batch_size = 32
        input_tensor = torch.randn((batch_size, self.model.in_features))
        output_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape, (batch_size, self.model.out_features))

    def test_num_layers(self):
        assert self.config.feedforward_layers
        self.assertEqual(len(self.model.sequence), len(self.config.feedforward_layers) * 3 - 1)
