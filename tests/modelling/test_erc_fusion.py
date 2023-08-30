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
        self.config = ERCConfig(classifier_classes=["a", "b"], fusion_attention_heads_degree=3, fusion_out_features=768)
        self.embedding_dims = [768, 768 * 2, 2048]
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
        expected_shape = (batch_size, self.config.fusion_out_features)
        self.assertEqual(output_tensor.shape, expected_shape)

    def test_out_features(self):
        self.assertEqual(self.fusion.out_features, self.config.fusion_out_features)

    def test_forward_not_normalized(self):
        batch_size = 3
        input_tensors = [torch.randn((batch_size, d)) for d in self.embedding_dims]
        output_tensor = self.fusion.forward(*input_tensors)
        norms = torch.norm(output_tensor, dim=1)
        for norm in norms:
            self.assertNotAlmostEqual(norm.item(), 1.0, places=5)

    def test_attn_in_features_does_not_match_concat_embeds_dims(self):
        self.assertNotEqual(self.fusion.attn.embed_dim, sum(self.embedding_dims))

    def test_attn_in_features_matches_fusion_out_features(self):
        self.assertEqual(self.fusion.attn.embed_dim, self.config.fusion_out_features)

    def test_num_heads_degree_2(self):
        self._test_num_heads_degree_n(degree=2, expected=2)

    def test_num_heads_degree_3(self):
        self._test_num_heads_degree_n(degree=3, expected=3)

    def test_num_heads_degree_4(self):
        self._test_num_heads_degree_n(degree=4, expected=4)

    def test_num_heads_degree_5(self):
        self._test_num_heads_degree_n(degree=5, expected=6)

    def test_num_heads_degree_6(self):
        self._test_num_heads_degree_n(degree=6, expected=8)

    def _test_num_heads_degree_n(self, degree, expected):
        config = ERCConfig(
            classifier_classes=self.config.classifier_classes,
            fusion_out_features=self.config.fusion_out_features,
            fusion_attention_heads_degree=degree,
        )
        fusion_type = ERCFusion.resolve_type_from(ERCFusionTechnique.MULTI_HEADED_ATTENTION)
        fusion_instance = fusion_type(self.embeddings, config=config)
        self.assertEqual(fusion_instance.attn.num_heads, expected)
