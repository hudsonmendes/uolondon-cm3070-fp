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

    def test_call(self):
        with wave.open("tests/fixtures/d-1038-seq-17.wav") as wave_file:
            result = self.preprocessor(wave_file)
            self.assertEqual(result.shape, (325458,))
