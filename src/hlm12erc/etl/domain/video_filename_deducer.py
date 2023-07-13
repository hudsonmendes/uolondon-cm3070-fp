import pandas as pd


class VideoFileNameDeducer:
    """
    Deduce the video filename from the given row.
    """

    def __call__(self, row: pd.Series) -> str:
        """
        Get the video filename from the given row.
        :param row: The row to get the video filename from.
        :return: The video filename.
        """
        return f"dia{row.dialogue}_utt{row.seq}.mp4"
