# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig, ERCFusionTechnique
from hlm12erc.modelling.erc_fusion import ERCEmbeddings, ERCStackedFusion


class TestERCStackedFusion(unittest.TestCase):
    def setUp(self):
        self.embedding_dims = [100, 200, 300]
        self.embeddings = [ERCEmbeddings(ERCConfig(d), d) for d in self.embedding_dims]
        self.config = ERCConfig(modules_fusion=ERCFusionTechnique.STACKED)
        self.fusion = ERCStackedFusion(self.embeddings, self.config)

    def tearDown(self):
        del self.fusion

    def test_forward_shape(self):
        input_tensors = [torch.randn((32, d)) for d in self.embedding_dims]
        output_tensor = self.fusion.forward(*input_tensors)
        expected_shape = (32, sum(self.embedding_dims))
        self.assertEqual(output_tensor.shape, expected_shape)

    def test_forward_output(self):
        input_tensors = [torch.randn((32, d)) for d in self.embedding_dims]
        output_tensor = self.fusion.forward(*input_tensors)
        expected_output = torch.cat(input_tensors, dim=1)
        self.assertTrue(torch.allclose(output_tensor, expected_output))

    def test_out_features(self):
        expected_out_features = sum(self.embedding_dims)
        self.assertEqual(self.fusion.out_features, expected_out_features)
