# Python Built-in Modules
import pathlib
import unittest

# My Packages and Modules
from hlm12erc.modelling import ERCConfig
from hlm12erc.training.meld_record_preprocessor_visual import (
    MeldVisualPreprocessorFilepathToResnet50,
)


class TestMeldVisualPreprocessor(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(classifier_classes=["a", "b", "c"])
        self.preprocessor = MeldVisualPreprocessorFilepathToResnet50()

    def tearDown(self):
        pass

    def test_call_creates_4d_tensor(self):
        result = self.preprocessor(pathlib.Path("tests/fixtures/d-1038-seq-17.png"))
        self.assertEqual(result.shape, self.config.visual_in_features)
