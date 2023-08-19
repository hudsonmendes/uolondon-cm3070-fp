# Python Built-in Modules
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_config import ERCConfig, ERCTextEmbeddingType
from hlm12erc.modelling.erc_emb_text import ERCTextEmbeddings


class TestERCGloveTextEmbeddings(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(text_in_features=50, text_out_features=50, classifier_classes=["a", "b"])
        self.embeddings = ERCTextEmbeddings.resolve_type_from(ERCTextEmbeddingType.GLOVE)(config=self.config)

    def test_forward_shape(self):
        input_list = ["here a test sentence", "this is another test sentence"]
        output_tensor = self.embeddings(input_list)
        self.assertEqual(output_tensor.shape, (len(input_list), self.config.text_out_features))

    def test_out_features(self):
        self.assertEqual(self.embeddings.out_features, self.config.text_out_features)

    def test_forward_oov_term_not_counted(self):
        tensor_without_oov = self.embeddings(["ones halves"])
        tensor_with_oov = self.embeddings(["ones ########################################################### halves"])
        self.assertEqual(tensor_without_oov.mean().item(), tensor_with_oov.mean().item())

    def test_forward_mean(self):
        input_list = ["ones halves"]
        output_tensor = self.embeddings(input_list)
        self.assertAlmostEqual(output_tensor.mean().item(), 0.02751464769244194, places=5)

    def test_forward_normalization(self):
        input_list = ["here a test sentence", "this is another test sentence"]
        output_tensor = self.embeddings(input_list)
        norms = torch.norm(output_tensor, dim=1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=5)


class TestERCGpt2TextEmbeddings(unittest.TestCase):
    def setUp(self):
        config = ERCConfig(
            modules_text_encoder=ERCTextEmbeddingType.GPT2,
            classifier_classes=["a", "b"],
            text_limit_to_n_last_tokens=4,
        )
        self.embeddings = ERCTextEmbeddings.resolve_type_from(config.modules_text_encoder)(config=config)

    def tearDown(self):
        del self.embeddings

    def test_forward_shape(self):
        input_list = ["here a test sentence", "this is another test sentence"]
        output_tensor = self.embeddings(input_list)
        self.assertEqual(output_tensor.shape, (len(input_list), 768))

    def test_out_features(self):
        self.assertEqual(self.embeddings.out_features, 768)

    def test_forward_normalization(self):
        input_list = ["here a test sentence", "this is another test sentence"]
        output_tensor = self.embeddings(input_list)
        norms = torch.norm(output_tensor, dim=1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_token_limitation(self):
        embedding1 = ERCTextEmbeddings.resolve_type_from(ERCTextEmbeddingType.GPT2)(
            config=ERCConfig(
                modules_text_encoder=ERCTextEmbeddingType.GPT2,
                classifier_classes=["a", "b"],
                text_limit_to_n_last_tokens=2,
            )
        )(["here a test sentence"])[0]
        embedding2 = ERCTextEmbeddings.resolve_type_from(ERCTextEmbeddingType.GPT2)(
            config=ERCConfig(
                modules_text_encoder=ERCTextEmbeddingType.GPT2,
                classifier_classes=["a", "b"],
                text_limit_to_n_last_tokens=3,
            )
        )(["here a test sentence"])[0]
        embedding3 = ERCTextEmbeddings.resolve_type_from(ERCTextEmbeddingType.GPT2)(
            config=ERCConfig(
                modules_text_encoder=ERCTextEmbeddingType.GPT2,
                classifier_classes=["a", "b"],
                text_limit_to_n_last_tokens=3,
            )
        )(["here a test sentence"])[0]
        self.assertNotEqual(embedding1.tolist(), embedding2.tolist())
        self.assertEqual(embedding2.tolist(), embedding3.tolist())
