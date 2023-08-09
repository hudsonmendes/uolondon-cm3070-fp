# Python Built-in Modules
import unittest

# Third-Party Libraries
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

    def test_call_creates_4d_tensor(self):
        with Image.open("tests/fixtures/d-1038-seq-17.png") as image:
            result = self.preprocessor(image)
            self.assertEqual(result.shape, self.config.visual_in_features)
