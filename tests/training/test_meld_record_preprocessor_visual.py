# Python Built-in Modules
import pathlib
import unittest

# Third-Party Libraries
import torch
from PIL.Image import Image

# My Packages and Modules
from hlm12erc.modelling import ERCConfig
from hlm12erc.training.meld_record_preprocessor_visual import (
    MeldVisualPreprocessorFilepathToFaceOnlyImage,
    MeldVisualPreprocessorFilepathToResnet50,
)


class TestMeldVisualPreprocessorFilepathToResnet50(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(classifier_classes=["a", "b", "c"])
        self.preprocessor = MeldVisualPreprocessorFilepathToResnet50()

    def tearDown(self):
        pass

    def test_call_creates_4d_tensor(self):
        result = self.preprocessor(pathlib.Path("tests/fixtures/d-1038-seq-17.png"))
        self.assertEqual(result.shape, self.config.visual_in_features)


class TestMeldVisualPreprocessorFilepathToFaceOnlyImage(unittest.TestCase):
    def setUp(self):
        self.encoder = MeldVisualPreprocessorFilepathToResnet50()
        self.preprocessor = MeldVisualPreprocessorFilepathToFaceOnlyImage(
            pathlib.Path(".weights_cache/retinaface_resnet50.pth")
        )

    def tearDown(self):
        pass

    def test_call_returns_image(self):
        result = self.preprocessor(pathlib.Path("tests/fixtures/d-1038-seq-17.png"))
        self.assertIsInstance(result, Image)

    def test_call_generates_embeddings_different_to_full_scene(self):
        filepath = pathlib.Path("tests/fixtures/d-1038-seq-17.png")
        preprocessed = self.preprocessor(filepath)
        embeddings_full = self.encoder(filepath)
        embeddings_full_another = self.encoder(filepath)
        embedding_faces = self.encoder(preprocessed)
        self.assertTrue(embeddings_full.tolist() == embeddings_full_another.tolist())
        self.assertFalse(embeddings_full.tolist() == embedding_faces.tolist())
