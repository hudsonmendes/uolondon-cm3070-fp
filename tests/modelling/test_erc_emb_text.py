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
        self.config = ERCConfig(text_in_features=50, text_out_features=50)
        self.embeddings = ERCTextEmbeddings.resolve_type_from(ERCTextEmbeddingType.GLOVE)(config=self.config)

    def tearDown(self):
        del self.embeddings
        mock.patch.stopall()

    def test_forward_shape(self):
        input_list = ["here a test sentence", "this is another test sentence"]
        output_tensor = self.embeddings(input_list)
        self.assertEqual(output_tensor.shape, (len(input_list), self.config.text_out_features))

    @mock.patch("hlm12erc.modelling.erc_emb_text.torchtext.vocab.GloVe.get_vecs_by_tokens")
    def test_forward_oov_are_not_counted(self, get_vecs_by_tokens_mock: mock.MagicMock):
        def get_vecs_by_tokens_fn(x, lower_case_backup: bool = False):
            return torch.ones((3,)) if x == "term" else torch.zeros((3,))

        get_vecs_by_tokens_mock.side_effect = get_vecs_by_tokens_fn
        input_list = ["term oov term"]
        output_tensor = self.embeddings(input_list)
        self.assertEqual(output_tensor.tolist(), [[1.0, 1.0, 1.0]])

    @mock.patch("hlm12erc.modelling.erc_emb_text.torchtext.vocab.GloVe.get_vecs_by_tokens")
    def test_forward_mean(self, get_vecs_by_tokens_mock: mock.MagicMock):
        def get_vecs_by_tokens_fn(x, lower_case_backup: bool = False):
            return torch.ones((3,)) if x == "ones" else (torch.ones((3,)) / 2)

        get_vecs_by_tokens_mock.side_effect = get_vecs_by_tokens_fn
        input_list = ["ones halves"]
        output_tensor = self.embeddings(input_list)
        self.assertEqual(output_tensor.tolist(), [[0.75, 0.75, 0.75]])
