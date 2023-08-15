# Python Built-in Modules
import unittest

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig
from hlm12erc.training.erc_config_formatter import ERCConfigFormatter


class TestERCConfig(unittest.TestCase):
    def test_str_classifier_name(self):
        config = ERCConfig(classifier_name="special_tag", classifier_classes=["a", "b", "c"])
        self.assertEqual(ERCConfigFormatter(config).represent(), "hlm12erc-special_tag")

    def test_str_feedforward_layers(self):
        config = ERCConfig(classifier_name="another_tag", classifier_classes=["a", "b", "c"])
        self.assertEqual(ERCConfigFormatter(config).represent(), "hlm12erc-another_tag")
