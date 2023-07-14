# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_emb_text import ERCGloveTextEmbeddings


class TestERCGloveTextEmbeddings(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embedding_dim = 50
        self.padding_idx = 0
        self.embeddings = ERCGloveTextEmbeddings(self.vocab_size, self.embedding_dim, self.padding_idx)

    def tearDown(self):
        del self.embeddings

    def test_forward_shape(self):
        input_list = ["this is a test sentence", "this is another test sentence"]
        output_tensor = self.embeddings(input_list)
        self.assertEqual(output_tensor.shape, (2, len(input_list[0].split()), self.embedding_dim))

    def test_forward_padding(self):
        input_list = ["this is a test sentence", "this is another test sentence"]
        output_tensor = self.embeddings(input_list)
        self.assertTrue(torch.all(torch.eq(output_tensor[0, -2:], torch.zeros((2, self.embedding_dim)))))

    def test_forward_out_of_vocab(self):
        input_list = [
            "this is a test sentence",
            "this is another test sentence",
            "this is a sentence with out-of-vocab words",
        ]
        output_tensor = self.embeddings(input_list)
        self.assertTrue(torch.all(torch.eq(output_tensor[2, -2:], torch.zeros((2, self.embedding_dim)))))

    def test_forward_grad(self):
        input_list = ["this is a test sentence", "this is another test sentence"]
        output_tensor = self.embeddings(input_list)
        loss = output_tensor.sum()
        loss.backward()
        self.assertIsNotNone(input_tensor.grad)
