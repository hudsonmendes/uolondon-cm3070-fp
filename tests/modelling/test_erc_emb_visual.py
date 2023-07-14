# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_emb_visual import (
    ERCConfig,
    ERCResNet50VisualEmbeddings,
    ERCVisualEmbeddings,
    ERCVisualEmbeddingType,
)


class TestERCResNet50VisualEmbeddings(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig()
        self.embeddings = ERCResNet50VisualEmbeddings(self.config)

    def tearDown(self):
        del self.embeddings

    def test_forward_shape(self):
        input_tensor = torch.randn((32, 3, 224, 224))
        output_tensor = self.embeddings(input_tensor)
        self.assertEqual(output_tensor.shape, (32, self.embeddings.out_features))

    def test_forward_grad(self):
        input_tensor = torch.randn((32, 3, 224, 224), requires_grad=True)
        output_tensor = self.embeddings(input_tensor)
        loss = output_tensor.sum()
        loss.backward()
        self.assertIsNotNone(input_tensor.grad)

    def test_out_features(self):
        self.assertEqual(self.embeddings.out_features, 2048)

    def test_resolve_type_from(self):
        self.assertEqual(
            ERCVisualEmbeddings.resolve_type_from(ERCVisualEmbeddingType.RESNET50), ERCResNet50VisualEmbeddings
        )

    def test_resolve_type_from_error(self):
        with self.assertRaises(ValueError):
            ERCVisualEmbeddings.resolve_type_from("invalid_type")
