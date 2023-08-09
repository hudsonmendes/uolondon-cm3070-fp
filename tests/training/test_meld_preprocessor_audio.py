# Python Built-in Modules
import unittest
import wave

# My Packages and Modules
from hlm12erc.training.meld_preprocessor_audio import MeldAudioPreprocessor


class TestMeldAudioPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = MeldAudioPreprocessor()

    def tearDown(self):
        pass

    def test_call_creates_tensor_for_audio_1_waveform(self):
        with wave.open("tests/fixtures/d-1038-seq-17.wav") as wave_file:
            result = self.preprocessor(wave_file)
            self.assertEqual(result.shape, (325458,))

    def test_call_creates_tensor_for_audio_2_waveform(self):
        with wave.open("tests/fixtures/dia1_utt0.wav") as wave_file:
            result = self.preprocessor(wave_file)
            self.assertEqual(result.shape, (218736,))
