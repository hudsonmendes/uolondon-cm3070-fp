# Python Built-in Modules
import unittest
from unittest.mock import MagicMock

# Third-Party Libraries
import pandas as pd

# My Packages and Modules
from hlm12erc.training.meld_record_preprocessor_text import MeldTextPreprocessor


class TestMeldTextPreprocessor(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "dialogue": [1, 1, 1, 2, 2, 2],
                "sequence": [1, 2, 3, 1, 2, 3],
                "speaker": ["A", "B", "A", "B", "A", "B"],
                "x_text": ["Hello", "Hi", "How are you?", "Goodbye", "See you later", "Bye"],
            }
        )
        self.preprocessor = MeldTextPreprocessor(self.df)

    def tearDown(self):
        pass

    def test_call_creates_dialog(self):
        # Arrange
        mock_row = MagicMock(spec=pd.Series)
        mock_row.dialogue = 1
        mock_row.sequence = 4
        mock_row.speaker = "A"
        mock_row.x_text = "Howdy"

        expected = "A said: Hello\nB said: Hi\nA said: How are you?\nA said: Howdy"

        # Act
        result = self.preprocessor(mock_row)

        # Assert
        self.assertEqual(result, expected)
