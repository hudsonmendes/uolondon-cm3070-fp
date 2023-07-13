import pathlib
from typing import Optional

import pandas as pd
from moviepy.editor import VideoFileClip
from PIL import Image


class VideoToImageMosaicTransformer:
    """
    Produce a mosaic of images from a video.
    """

    dest: pathlib.Path
    n: Optional[int] = None

    def __init__(self, dest: pathlib.Path, n: Optional[int] = None) -> None:
        """
        Create a new mosaic producer that produces a mosaic of images
        from a video.
        :param dest: The destination directory to save the mosaic image to.
        :param n: The number of screenshots to extract from the video.
        """
        self.dest = dest
        self.n = n

    def __call__(self, row: pd.Series) -> str:
        """
        Extracts a number of screenshots defined by `self.n` from the original
        .mp4 video, equidistant to one another.

        :param filepath: The filepath to the video to extract the screenshots from.
        :param dialogue: The dialogue ID to use in the mosaic filename.
        :param utterance_id: The utterance ID to use in the mosaic filename.
        :return: The filepath of the extracted screenshots mosaic image.
        """
        # open the video file to extract the screenshots
        filename = row["x_av"]
        clip = VideoFileClip(filename)

        # find the duration in seconds of the video clip
        duration = clip.duration

        # calculate the timestamps for the screenshots
        n_screenshots = self.n if (self.n and self.n > 0) else 3
        timestamps = [duration * i / (n_screenshots - 1) for i in range(n_screenshots)]

        # extract the screenshots and stack them on top of each other
        screenshots = []
        for timestamp in timestamps:
            screenshot = clip.get_frame(timestamp)
            screenshots.append(Image.fromarray(screenshot))

        mosaic = Image.new("RGB", (screenshots[0].width, screenshots[0].height * n_screenshots))
        for i, screenshot in enumerate(screenshots):
            mosaic.paste(screenshot, (0, i * screenshot.height))

        # save the mosaic image to the destination directory with the specified filename
        filename = f"d-{row['dialogue']}-seq-{row['seq']}.png"
        filepath = self.dest / filename
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)
        mosaic.save(filepath)
        return filename
