import unittest

import pandas as pd

from hlm12erc.etl.domain.video_filename_deducer import VideoFileNameDeducer


class TestVideoFilenameDeducer(unittest.TestCase):
    def setUp(self):
        self.deducer = VideoFileNameDeducer()

    def test_dialogue_number(self):
        row = pd.Series({"dialogue": 1, "seq": 1})
        self.assertEqual(self.deducer(row), "dia1_utt1.mp4")

    def test_sequence_number(self):
        row = pd.Series({"dialogue": 1, "seq": 2})
        self.assertEqual(self.deducer(row), "dia1_utt2.mp4")

    def test_large_dialogue_number(self):
        row = pd.Series({"dialogue": 1234, "seq": 1})
        self.assertEqual(self.deducer(row), "dia1234_utt1.mp4")

    def test_large_sequence_number(self):
        row = pd.Series({"dialogue": 1, "seq": 1234})
        self.assertEqual(self.deducer(row), "dia1_utt1234.mp4")

    def test_zero_dialogue_number(self):
        row = pd.Series({"dialogue": 0, "seq": 1})
        self.assertEqual(self.deducer(row), "dia0_utt1.mp4")

    def test_zero_sequence_number(self):
        row = pd.Series({"dialogue": 1, "seq": 0})
        self.assertEqual(self.deducer(row), "dia1_utt0.mp4")
