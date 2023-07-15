# Python Built-in Modules
import unittest
import wave

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_emb_audio import ERCAudioEmbeddings, ERCAudioEmbeddingType, ERCConfig


class TestERCRawAudioEmbeddings(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(audio_in_features=325458, audio_out_features=512)
        self.embeddings = ERCAudioEmbeddings.resolve_type_from(ERCAudioEmbeddingType.WAVEFORM)(self.config)
        self.audios = [
            wave.open("tests/fixtures/d-1038-seq-17.wav"),
            wave.open("tests/fixtures/dia1_utt0.wav"),
        ]

    def tearDown(self):
        del self.embeddings

    def test_out_features(self):
        self.assertEqual(self.embeddings.out_features, 512)

    def test_forward_shape(self):
        output_tensor = self.embeddings(self.audios)
        self.assertEqual(output_tensor.shape, (len(self.audios), self.embeddings.out_features))

    def test_forward_normalization(self):
        output_tensor = self.embeddings(self.audios)
        norms = torch.norm(output_tensor, dim=1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=5)
