# Python Built-in Modules
import unittest
import unittest.mock

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_emb import ERCEmbeddings
from hlm12erc.modelling.erc_fusion import ERCConfig, ERCFusion, ERCFusionTechnique


class TestERCConcatFusion(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(classifier_classes=["a", "b"])
        self.embedding_dims = [100, 200, 300]
        self.embeddings = [unittest.mock.create_autospec(ERCEmbeddings, out_features=f) for f in self.embedding_dims]
        self.fusion = ERCFusion.resolve_type_from(ERCFusionTechnique.CONCATENATION)(
            embeddings=self.embeddings,
            config=self.config,
        )

    def tearDown(self):
        del self.fusion

    def test_forward_shape(self):
        batch_size = 3
        input_tensors = [torch.randn((batch_size, d)) for d in self.embedding_dims]
        output_tensor = self.fusion.forward(*input_tensors)
        expected_shape = (batch_size, sum(self.embedding_dims))
        self.assertEqual(output_tensor.shape, expected_shape)

    def test_out_features(self):
        self.assertEqual(self.fusion.out_features, sum(self.embedding_dims))

    def test_forward_normalization(self):
        batch_size = 3
        input_tensors = [torch.randn((batch_size, d)) for d in self.embedding_dims]
        output_tensor = self.fusion.forward(*input_tensors)
        norms = torch.norm(output_tensor, dim=1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=5)


class TestERCMultiheadedAttentionFusion(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(classifier_classes=["a", "b"], fusion_attention_heads_degree=3)
        self.embedding_dims = [100, 200, 300]
        self.embeddings = [unittest.mock.create_autospec(ERCEmbeddings, out_features=f) for f in self.embedding_dims]
        self.fusion = ERCFusion.resolve_type_from(ERCFusionTechnique.MULTI_HEADED_ATTENTION)(
            embeddings=self.embeddings,
            config=self.config,
        )

    def tearDown(self):
        del self.fusion

    def test_forward_shape(self):
        batch_size = 3
        input_tensors = [torch.randn((batch_size, d)) for d in self.embedding_dims]
        output_tensor = self.fusion.forward(*input_tensors)
        expected_shape = (batch_size, sum(self.embedding_dims))
        self.assertEqual(output_tensor.shape, expected_shape)

    def test_out_features(self):
        self.assertEqual(self.fusion.out_features, sum(self.embedding_dims))

    def test_forward_not_normalization(self):
        batch_size = 3
        input_tensors = [torch.randn((batch_size, d)) for d in self.embedding_dims]
        output_tensor = self.fusion.forward(*input_tensors)
        norms = torch.norm(output_tensor, dim=1)
        for norm in norms:
            self.assertNotAlmostEqual(norm.item(), 1.0, places=5)
