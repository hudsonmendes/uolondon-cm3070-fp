# Python Built-in Modules
import pathlib
import unittest

# Third-Party Libraries
import torch

# My Packages and Modules
from hlm12erc.training.meld_dataset import MeldDataset
from hlm12erc.training.meld_record import MeldRecord
from hlm12erc.training.meld_record_preprocessor_visual import MeldVisualPreprocessorFilepathToFaceOnlyImage


class TestMeldRecordReader(unittest.TestCase):
    def setUp(self):
        self.dataset = MeldDataset(filepath=pathlib.Path("tests/fixtures/sample.csv"))

    def test_getitem_out_of_range(self):
        with self.assertRaises(IndexError):
            self.dataset[len(self.dataset)]

    def test_getitem_within_range(self):
        actual = self.dataset[0]
        self.assertIsNotNone(actual)
        self.assertIsInstance(actual, MeldRecord)

    def test_getitem_audio_waveform(self):
        actual = self.dataset[0]
        self.assertIsInstance(actual.audio, torch.Tensor)
        self.assertEqual(actual.audio.shape, (325458,))

    def test_getitem_text_prompt_tail(self):
        actual = self.dataset[0]
        self.assertEqual(
            actual.text,
            'The speaker "Chandler" said:\n###\nalso I was the point person on my companys transition from the KL-5 to GR-6 system.\n###',
        )

    def test_getitem_text_head_at_position_2(self):
        actual = self.dataset[1]
        self.assertEqual(
            actual.text,
            "\n\n\n".join(
                [
                    'The speaker "Chandler" said:\n###\nalso I was the point person on my companys transition from the KL-5 to GR-6 system.\n###',
                    'The speaker "Chandler" said:\n###\ncool!\n###',
                ]
            ),
        )

    def test_getitem_visual_shape(self):
        actual = self.dataset[0]
        self.assertIsInstance(actual.visual, torch.Tensor)
        self.assertEqual(actual.visual.shape, (3, 256, 721))

    def test_getitem_audio_shape(self):
        actual = self.dataset[0]
        self.assertIsInstance(actual.audio, torch.Tensor)
        self.assertEqual(actual.audio.shape, (325458,))

    def test_getitem_with_additional_visual_preprocessing(self):
        pp = MeldVisualPreprocessorFilepathToFaceOnlyImage(pathlib.Path(".weights_cache/retinaface_resnet50.pth"))
        actual = self.dataset.preprocessing_with(pp)[0]
        self.assertIsInstance(actual.visual, torch.Tensor)
        self.assertEqual(actual.visual.shape, (3, 256, 721))
