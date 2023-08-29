# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch
from PIL import Image

# My Packages and Modules
from hlm12erc.modelling.erc_emb_visual import (
    ERCConfig,
    ERCVisualEmbeddings,
    ERCVisualEmbeddingType,
)
from hlm12erc.training.erc_data_collator import ERCDataCollator
from hlm12erc.training.meld_record_preprocessor_visual import MeldVisualPreprocessor


class TestERCResNet50VisualEmbeddings(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(classifier_classes=["a", "b"])
        self.embeddings = ERCVisualEmbeddings.resolve_type_from(ERCVisualEmbeddingType.RESNET50)(self.config)
        self.preprocessor = MeldVisualPreprocessor()
        self.data_collator = ERCDataCollator(config=self.config, label_encoder=None)
        self.images = self.data_collator._visual_to_stacked_tensor(
            [self.preprocessor(Image.open("tests/fixtures/d-1038-seq-17.png"))]
        )

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
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=5)
