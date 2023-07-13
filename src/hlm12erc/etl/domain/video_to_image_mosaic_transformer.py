# Python Built-in Modules
import logging
import math
import pathlib
import warnings
from typing import Optional

# Third-Party Libraries
import pandas as pd
from moviepy.editor import VideoFileClip
from PIL import Image

logger = logging.getLogger(__name__)


class VideoToImageMosaicTransformer:
    """
    Produce a mosaic of images from a video.
    """

    dest: pathlib.Path
    n: Optional[int] = None
    height: int = 480

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
        # suppress warnings from moviepy
        self._suppress_warnings()

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

        # extract the screenshots and stack them on top of each other
        screenshots = []
        for timestamp in timestamps:
            screenshots.append(self.extract_screenshot_at(clip, timestamp))

        mosaic = Image.new("RGB", (screenshots[0].width, screenshots[0].height * n_screenshots))
        for i, screenshot in enumerate(screenshots):
            mosaic.paste(screenshot, (0, i * screenshot.height))

        # save the mosaic image to the destination directory with the specified filename
        filename = f"d-{row.dialogue}-seq-{row.sequence}.png"
        filepath = self.dest / filename
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)
        mosaic.save(filepath)
        return filename

    def extract_screenshot_at(self, clip, timestamp) -> Image.Image:
        """
        Attempts to extract a screeshot from the video clip at the specified
        timestamp. If the timestamp is invalid, the screenshot at the start of
        the video clip is returned instead. And if the video clip is invalid,
        a blank image is returned instead.

        :param clip: The video clip to extract the screenshot from.
        :param timestamp: The timestamp to extract the screenshot at.
        :return: The screenshot at the specified timestamp.
        """
        height = self.height
        try:
            # extract and resize the screenshot
            screenshot = clip.get_frame(timestamp)
            screenshot = Image.fromarray(screenshot)
            width = math.floor(screenshot.width * height / screenshot.height)
            screenshot = screenshot.resize((width, height))
        except Exception as e:
            # create dummy screenshot if the video clip is invalid
            # the mock width 853 has been chosen as it is the width of
            # the frames extracted from some actual video clips
            logger.warning(f"Failed to extract screenshot at {timestamp} from {clip.filename}: {e}")
            screenshot = Image.new("RGB", (self.height, 853))
        return screenshot

    def _suppress_warnings(self) -> None:
        """
        Uses a custom warning filter to suppress the warnings from the moviepy
        library which is likely to produce multiple warnings and fallback to
        acceptable behaviour
        """

        # define a filter function to suppress warnings from the moviepy library
        def moviepy_warning_filter(message, category, filename, lineno, file=None, line=None):
            if category == UserWarning and "moviepy" in str(filename):
                return None
            else:
                return message, category, filename, lineno, file, line

        # add the filter function to the warnings registry
        warnings.showwarning = moviepy_warning_filter
