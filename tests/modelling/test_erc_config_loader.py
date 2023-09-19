# Python Built-in Modules
import pathlib
import unittest

# My Packages and Modules
from hlm12erc.modelling.erc_config_loader import ERCConfigLoader


class TestErcConfigLoader(unittest.TestCase):
    def test_unimodal_config_with_missing_keys(self):
        actual = ERCConfigLoader(pathlib.Path("tests/fixtures/t-um-glove.yml")).load()
        self.assertEqual(actual.modules_text_encoder, "glove")
        self.assertIsNone(actual.visual_in_features)
        self.assertIsNone(actual.audio_in_features)

    def test_unimodal_config_with_null_values(self):
        actual = ERCConfigLoader(pathlib.Path("tests/fixtures/t-um-glove.json")).load()
        self.assertEqual(actual.modules_text_encoder, "glove")
        self.assertIsNone(actual.visual_in_features)
        self.assertIsNone(actual.audio_in_features)
