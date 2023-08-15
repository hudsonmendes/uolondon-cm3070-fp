# Python Built-in Modules
import pathlib
import unittest

# Third-Party Libraries
import pandas as pd
import torch
from PIL import Image

# My Packages and Modules
from hlm12erc.training.meld_record import MeldRecord
from hlm12erc.training.meld_record_reader import MeldRecordReader


class TestMeldRecordReader(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "dialogue": [1, 1],
                "sequence": [1, 2],
                "speaker": ["Ross", "Rachel"],
                "x_text": ["i think we should do it this way", "yeah?"],
                "x_audio": ["d-1038-seq-17.wav", "d-1038-seq-17.wav"],
                "x_visual": ["d-1038-seq-17.png", "d-1038-seq-17.png"],
                "label": [2, 3],
            }
        )
        self.reader = MeldRecordReader(filename="test.csv", filedir=pathlib.Path("tests/fixtures"), df=self.df)
        self.result = self.reader.read_at(0)

    def test_read_at_0_returns_none_if_index_out_of_range(self):
        result = self.reader.read_at(len(self.df))
        self.assertIsNone(result)

    def test_read_at_0_returns_meld_record(self):
        self.assertIsNotNone(self.result)
        self.assertIsInstance(self.result, MeldRecord)

    def test_read_at_0_returns_meld_record_with_audio(self):
        self.assertIsInstance(self.result.audio, torch.Tensor)
        self.assertEqual(self.result.audio.shape, (325458,))

    def test_read_at_0_returns_meld_record_with_correct_text(self):
        self.assertEqual(self.result.text, 'The speaker "Ross" said:\n###\ni think we should do it this way\n###')

    def test_read_at_1_returns_meld_record_with_correct_text_with_dialog(self):
        result = self.reader.read_at(1)
        self.assertEqual(
            result.text,
            "\n\n\n".join(
                [
                    'The speaker "Ross" said:\n###\ni think we should do it this way\n###',
                    'The speaker "Rachel" said:\n###\nyeah?\n###',
                ]
            ),
        )

    def test_read_at_0_returns_meld_record_with_correct_visual(self):
        self.assertIsInstance(self.result.visual, torch.Tensor)
        self.assertEqual(self.result.visual.shape, (3, 256, 721))

    def test_read_at_0_returns_meld_record_with_correct_audio(self):
        self.assertIsInstance(self.result.audio, torch.Tensor)
        self.assertEqual(self.result.audio.shape, (325458,))
