# Python Built-in Modules
import unittest
from unittest import mock

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig, ERCTextEmbeddingType
from hlm12erc.modelling.erc_emb_text import ERCTextEmbeddings


class TestERCGloveTextEmbeddings(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(text_out_features=10)
        self.embeddings = ERCTextEmbeddings.resolve_type_from(ERCTextEmbeddingType.GLOVE)(config=self.config)

    def tearDown(self):
        del self.embeddings
        mock.patch.stopall()

    def test_forward_shape(self):
        input_list = ["this is a test sentence", "this is another test sentence"]
        output_tensor = self.embeddings(input_list)
        self.assertEqual(output_tensor.shape, (len(input_list), self.config.text_out_features))

    def test_forward_oov(self):
        input_list = ["this is a test oov"]
        output_tensor = self.embeddings(input_list)
        self.assertEqual(output_tensor.shape, (len(input_list), self.config.text_out_features))

    @mock.patch("hlm12erc.modelling.erc_emb_text.torchtext.vocab.Glove.get_vecs_by_tokens")
    def test_forward_mean(self, get_vecs_by_tokens_mock):
        get_vecs_by_tokens_mock.behave = lambda x: torch.ones((3,)) if x == "one" else torch.zeros((3,))
        input_list = ["ones zeros"]
        output_tensor = self.embeddings(input_list)
        self.assertEqual(output_tensor.tolist(), [[0.5, 0.5, 0.5]])
