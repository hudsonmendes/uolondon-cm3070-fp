# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_feedforward import ERCFeedForwardActivation, ERCFeedForwardConfig, ERCFeedForwardModel


class TestERCFeedForwardModel(unittest.TestCase):
    def setUp(self):
        self.in_features = 100
        self.hidden_size = 50
        self.num_layers = 2
        self.dropout = 0.1
        self.activation = ERCFeedForwardActivation.RELU
        self.config = ERCFeedForwardConfig(self.hidden_size, self.num_layers, self.dropout, self.activation)
        self.model = ERCFeedForwardModel(self.in_features, self.config)

    def tearDown(self):
        del self.model

    def test_forward_shape(self):
        input_tensor = torch.randn((32, self.in_features))
        output_tensor = self.model(input_tensor)
        self.assertEqual(output_tensor.shape, (32, self.hidden_size))

    def test_forward_grad(self):
        input_tensor = torch.randn((32, self.in_features), requires_grad=True)
        output_tensor = self.model(input_tensor)
        loss = output_tensor.sum()
        loss.backward()
        self.assertIsNotNone(input_tensor.grad)

    def test_num_layers(self):
        self.assertEqual(len(self.model.ff), self.num_layers * 2 - 1)

    def test_activation(self):
        activation_type = ERCFeedForwardModel.resolve_activation_from(self.activation)
        self.assertIsInstance(self.model.ff[1], activation_type)

    def test_dropout(self):
        if self.dropout > 0:
            self.assertIsInstance(self.model.ff[2], torch.nn.Dropout)
        else:
            self.assertNotIsInstance(self.model.ff[2], torch.nn.Dropout)
