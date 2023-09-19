# Python Built-in Modules
import unittest
import wave

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.modelling.erc_emb_audio import ERCAudioEmbeddings, ERCAudioEmbeddingType, ERCConfig
from hlm12erc.training.erc_data_collator import ERCDataCollator
from hlm12erc.training.meld_record_preprocessor_audio import MeldAudioPreprocessorToWaveform


class TestERCRawAudioEmbeddings(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(audio_in_features=325458, audio_out_features=256, classifier_classes=["a", "b"])
        self.data_collator = ERCDataCollator(config=self.config, label_encoder=None)
        preprocessor = MeldAudioPreprocessorToWaveform()
        self.audios = self.data_collator._audio_to_stacked_tensor(
            [
                preprocessor(wave.open("tests/fixtures/d-1038-seq-17.wav")),
                preprocessor(wave.open("tests/fixtures/dia1_utt0.wav")),
            ]
        )

    def test_out_features(self):
        self.assertEqual(self._create_embeddings().out_features, self.config.audio_out_features)

    def test_forward_nonmha_shape(self):
        embeddings = self._create_embeddings(audio_waveform_attention_heads_degree=None)
        output_tensor = embeddings(self.audios)
        self.assertEqual(output_tensor.shape, (len(self.audios), embeddings.out_features))

    def test_forward_mha_shape(self):
        embeddings = self._create_embeddings(audio_waveform_attention_heads_degree=2)
        output_tensor = embeddings(self.audios)
        self.assertEqual(output_tensor.shape, (len(self.audios), embeddings.out_features))

    def test_forward_nonmha_normalization(self):
        embeddings = self._create_embeddings(audio_waveform_attention_heads_degree=None)
        output_tensor = embeddings(self.audios)
        norms = torch.norm(output_tensor, dim=1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_forward_mha_normalization(self):
        embeddings = self._create_embeddings(audio_waveform_attention_heads_degree=3)
        output_tensor = embeddings(self.audios)
        norms = torch.norm(output_tensor, dim=1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_forward_nonmha_non_normalization_when_l2norm_disabled(self):
        embeddings = self._create_embeddings(audio_waveform_attention_heads_degree=None, audio_l2norm=False)
        output_tensor = embeddings(self.audios)
        norms = torch.norm(output_tensor, dim=1)
        for norm in norms:
            self.assertNotAlmostEqual(norm.item(), 1.0, places=5)

    def test_forward_non_normalization_when_l2norm_disabled(self):
        embeddings = self._create_embeddings(audio_waveform_attention_heads_degree=3, audio_l2norm=False)
        output_tensor = embeddings(self.audios)
        norms = torch.norm(output_tensor, dim=1)
        for norm in norms:
            self.assertNotAlmostEqual(norm.item(), 1.0, places=5)

    def _create_embeddings(self, **kwargs) -> ERCAudioEmbeddings:
        config_dict = self.config.to_dict()
        config_dict.update(kwargs)
        config = ERCConfig(**config_dict)
        return ERCAudioEmbeddings.resolve_type_from(ERCAudioEmbeddingType.WAVEFORM)(config)


class TestERCWave2Vec2Embeddings(unittest.TestCase):
    def setUp(self):
        self.config = ERCConfig(audio_in_features=30000, audio_out_features=32, classifier_classes=["a", "b"])
        self.embeddings = ERCAudioEmbeddings.resolve_type_from(ERCAudioEmbeddingType.WAV2VEC2)(self.config)
        self.data_collator = ERCDataCollator(config=self.config, label_encoder=None)
        preprocessor = MeldAudioPreprocessorToWaveform()
        self.audios = self.data_collator._audio_to_stacked_tensor(
            [
                preprocessor(wave.open("tests/fixtures/d-1038-seq-17.wav")),
                preprocessor(wave.open("tests/fixtures/dia1_utt0.wav")),
            ]
        )

    def tearDown(self):
        del self.embeddings

    def test_out_features(self):
        self.assertEqual(self.embeddings.out_features, self.config.audio_out_features)

    def test_forward_shape_using_projection(self):
        output_tensor = self.embeddings(self.audios)
        self.assertEqual(output_tensor.shape, (len(self.audios), self.embeddings.out_features))

    def test_forward_shape_without_projection(self):
        config = ERCConfig(audio_in_features=16000, audio_out_features=-1, classifier_classes=["a", "b"])
        embeddings = ERCAudioEmbeddings.resolve_type_from(ERCAudioEmbeddingType.WAV2VEC2)(config)
        output_tensor = embeddings(self.audios)
        self.assertEqual(output_tensor.shape, (len(self.audios), embeddings.wav2vec2.config.hidden_size * 2))

    def test_forward_normalization(self):
        output_tensor = self.embeddings(self.audios)
        norms = torch.norm(output_tensor, dim=1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_forward_not_normalised_when_l2norm_disabled(self):
        config = ERCConfig(audio_in_features=100, audio_out_features=32, audio_l2norm=False, classifier_classes=["a"])
        embeddings = ERCAudioEmbeddings.resolve_type_from(ERCAudioEmbeddingType.WAV2VEC2)(config)
        output_tensor = embeddings(self.audios)
        norms = torch.norm(output_tensor, dim=1)
        for norm in norms:
            self.assertNotAlmostEqual(norm.item(), 1.0, places=5)
        del embeddings
