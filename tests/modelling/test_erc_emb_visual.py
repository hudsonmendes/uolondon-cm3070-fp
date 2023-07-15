# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch
from PIL import Image

# My Packages and Modules
from hlm12erc.modelling.erc_emb_visual import ERCConfig, ERCVisualEmbeddings, ERCVisualEmbeddingType


class TestERCResNet50VisualEmbeddings(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig()
        self.embeddings = ERCVisualEmbeddings.resolve_type_from(ERCVisualEmbeddingType.RESNET50)(self.config)
        self.images = [Image.open("tests/fixtures/d-1038-seq-17.png")]

    def tearDown(self):
        del self.embeddings

    def test_out_features(self):
        self.assertEqual(self.embeddings.out_features, 2048)

    def test_forward_shape(self):
        output_tensor = self.embeddings(self.images)
        self.assertEqual(output_tensor.shape, (len(self.images), self.embeddings.out_features))

    def test_forward_normalization(self):
        output_tensor = self.embeddings(self.images)
        norms = torch.norm(output_tensor, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms)))
