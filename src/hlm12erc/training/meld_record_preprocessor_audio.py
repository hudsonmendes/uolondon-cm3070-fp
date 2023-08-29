# Python Built-in Modules
import pathlib
import wave
from abc import ABC, abstractmethod

# Third-Party Libraries
import torch


class MeldAudioPreprocessor(ABC):
    """
    Abstract class that define the contract of audio preprocessors.
    """

    @abstractmethod
    def __call__(self, x: pathlib.Path | wave.Wave_read) -> wave.Wave_read | torch.Tensor:
        """
        when implemented, preprocesses either a filepath or an audio into
        either an audio or a tensor. Returning tensors should be the last
        step of the chain.

        :param x: The audio to be preprocessed, either a path or an audio
        :return: The preprocessed audio or the final tensor
        """
        raise NotImplementedError("Not yet implemented")


class MeldAudioPreprocessorToWaveform(MeldAudioPreprocessor):
    """
    Preprocessor class for the audio files, turning them into tensors,
    for better compatibility with TPU traininga without affecting negatively
    CPU-based training.
    """

    def __init__(self):
        """
        Creates a new instance of the MeldAudioPreprocessor class with the
        the `dtype_map` instantiated
        """
        self.dtype_map = {
            1: torch.int8,
            2: torch.int16,
            4: torch.int32,
        }

    def __call__(self, x: pathlib.Path | wave.Wave_read) -> wave.Wave_read | torch.Tensor:
        """
        Preprocesses the audio by converting it to a tensor.

        :param x: The audio to be preprocessed
        :return: The preprocessed audio, no padding or truncation applied
        """
        # ensure that we have a filepath to start with
        if isinstance(x, torch.Tensor):
            raise ValueError(
                """
                The input `x` is already a `torch.Tensor`, which means that the processor
                has already materialised the data into the format that the model will process
                and does not allow for preprocessing anymore."""
            )
        elif isinstance(x, pathlib.Path):
            x = wave.open(x, "rb")

        # we will turn the waveform into a tensor, therefore sealing it
        with x:
            data = x.readframes(x.getnframes())
            dtype = self.dtype_map[x.getsampwidth()]
            y = torch.frombuffer(data, dtype=dtype).clone().float()
            y = y.reshape(-1, x.getnchannels())
            y = y.flatten()
            return y
