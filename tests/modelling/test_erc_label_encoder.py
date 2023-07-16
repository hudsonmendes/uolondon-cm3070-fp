# Python Built-in Modules
import unittest

# My Packages and Modules
from hlm12erc.modelling.erc_label_encoder import ERCLabelEncoder


class TestERCLabelEncoder(unittest.TestCase):
    def setUp(self):
        self.classes = ["cat", "dog", "bird"]
        self.encoder = ERCLabelEncoder(self.classes)

    def tearDown(self):
        pass

    def test_classes_type(self):
        self.assertIsInstance(self.encoder.classes, list)

    def test_classes_length(self):
        self.assertEqual(len(self.encoder.classes), len(self.classes))

    def test_classes_content(self):
        self.assertListEqual(self.encoder.classes, sorted(self.classes))

    def test_encode_single(self):
        # rembemer that the labeller sorts the classes alphabetically
        label = "cat"
        expected = [0.0, 1.0, 0.0]
        self.assertListEqual(self.encoder.encode(label).tolist(), expected)

    def test_encode_multiple(self):
        # rembemer that the labeller sorts the classes alphabetically
        labels = ["cat", "dog", "bird"]
        expected = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        self.assertListEqual(self.encoder.encode(labels).tolist(), expected)

    def test_callable(self):
        self.assertTrue(callable(self.encoder))
