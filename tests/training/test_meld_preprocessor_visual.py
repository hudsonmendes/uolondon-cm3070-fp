# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch
from PIL import Image

# My Packages and Modules
from hlm12erc.modelling import ERCConfig
from hlm12erc.training.meld_preprocessor_visual import MeldVisualPreprocessor


class TestMeldVisualPreprocessor(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(classifier_classes=["a", "b", "c"])
        self.preprocessor = MeldVisualPreprocessor()

    def tearDown(self):
        pass

    def test_call(self):
        with Image.open("tests/fixtures/d-1038-seq-17.png") as image:
            result = self.preprocessor(image)
            self.assertEqual(result.shape, (3, 256, 721))
