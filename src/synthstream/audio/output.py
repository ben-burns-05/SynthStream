"""Output sinks shared by offline and live rendering paths."""

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import sounddevice as sd  # type: ignore[import-untyped]
import soundfile as sf  # type: ignore[import-untyped]

AudioSamples = npt.NDArray[np.float32]


class AudioSink(Protocol):
    """A destination that accepts a complete mono waveform."""

    def write(self, samples: AudioSamples, sample_rate: int) -> None:
        """Consume audio at the supplied sample rate."""
        ...


class WavFileSink:
    """Write rendered audio to a WAV file."""

    def __init__(self, path: str | Path, *, subtype: str = "PCM_16") -> None:
        self.path = Path(path)
        self.subtype = subtype

    def write(self, samples: AudioSamples, sample_rate: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(self.path, samples, sample_rate, subtype=self.subtype, format="WAV")


class SoundDeviceSink:
    """Play complete rendered audio through a selected physical output device."""

    def __init__(self, device: int | str | None = None) -> None:
        self.device = device

    def write(self, samples: AudioSamples, sample_rate: int) -> None:
        sd.play(samples, samplerate=sample_rate, device=self.device, blocking=True)

