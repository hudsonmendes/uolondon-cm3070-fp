# Python Built-in Modules
import pathlib
import shutil
import unittest
from unittest.mock import patch

# Third-Party Libraries
import pandas as pd

# My Packages and Modules
from hlm12erc.etl import RawTo1NFTransformer


class TestRawTo1NFTransformer(unittest.TestCase):
    def setUp(self):
        self.src = pathlib.Path("/tmp/hlm12erc/tests/test_data/")
        self.src.mkdir(parents=True, exist_ok=True)
        (self.src / "train_sent_emo.csv").touch()
        (self.src / "dev_sent_emo.csv").touch()
        (self.src / "test_sent_emo.csv").touch()
        self.workspace = pathlib.Path("/tmp/hlm12erc/tests/test_work/")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.dest = pathlib.Path("/tmp/hlm12erc/tests/test_output/")
        self.dest.mkdir(parents=True, exist_ok=True)
        self.transformer = RawTo1NFTransformer(self.src, self.workspace)

    def tearDown(self):
        del self.transformer
        shutil.rmtree(self.src)
        shutil.rmtree(self.workspace)
        shutil.rmtree(self.dest)

    @patch("hlm12erc.etl.RawTo1NFTransformer._collect_raw")
    @patch("hlm12erc.etl.RawTo1NFTransformer._collect_transformed")
    @patch("hlm12erc.etl.RawTo1NFTransformer._collect_wrapped")
    def test_transform_saves_csv_for_each_split(self, mock_collect_wrapped, mock_collect_transformed, mock_collect_raw):
        mock_collect_raw.return_value = pd.DataFrame()
        mock_collect_transformed.return_value = pd.DataFrame()
        mock_collect_wrapped.return_value = pd.DataFrame()

        self.transformer.transform(self.dest, force=False)

        self.assertTrue((self.dest / "train.csv").exists())
        self.assertTrue((self.dest / "valid.csv").exists())
        self.assertTrue((self.dest / "test.csv").exists())

    @patch("hlm12erc.etl.RawTo1NFTransformer._collect_raw")
    @patch("hlm12erc.etl.RawTo1NFTransformer._collect_transformed")
    @patch("hlm12erc.etl.RawTo1NFTransformer._collect_wrapped")
    def test_transform_saves_csv_for_each_split_with_force(
        self, mock_collect_wrapped, mock_collect_transformed, mock_collect_raw
    ):
        mock_collect_raw.return_value = pd.DataFrame()
        mock_collect_transformed.return_value = pd.DataFrame()
        mock_collect_wrapped.return_value = pd.DataFrame()

        self.transformer.transform(self.dest, force=True)

        self.assertTrue((self.dest / "train.csv").exists())
        self.assertTrue((self.dest / "valid.csv").exists())
        self.assertTrue((self.dest / "test.csv").exists())
