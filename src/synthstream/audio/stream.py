"""Realtime audio transport with production and fake duplex backends."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import Protocol

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]

from synthstream.audio.buffer import AudioRingBuffer
from synthstream.audio.output import AudioSamples

DuplexCallback = Callable[[AudioSamples, str | None], AudioSamples]


class DuplexAudioBackend(Protocol):
    """Backend contract used by physical and fake duplex devices."""

    def start(
        self, sample_rate: int, block_size: int, callback: DuplexCallback
    ) -> None:
        """Start delivering fixed-size input blocks to ``callback``."""
        ...

    def stop(self) -> None:
        """Stop callback delivery and release device resources."""
        ...


@dataclass(frozen=True, slots=True)
class StreamStatistics:
    """Snapshot of observable transport failures and buffer pressure."""

    input_overflow_samples: int
    output_overflow_samples: int
    output_underflow_samples: int
    callback_statuses: tuple[str, ...]


class SoundDeviceDuplexBackend:
    """Duplex PortAudio backend with independently selectable devices."""

    def __init__(
        self,
        *,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
    ) -> None:
        self.input_device = input_device
        self.output_device = output_device
        self._stream: object | None = None

    def start(
        self, sample_rate: int, block_size: int, callback: DuplexCallback
    ) -> None:
        if self._stream is not None:
            raise RuntimeError("audio backend is already running")

        def device_callback(
            indata: np.ndarray,
            outdata: np.ndarray,
            frames: int,
            time_info: object,
            status: object,
        ) -> None:
            del time_info
            input_samples = np.asarray(indata[:, 0], dtype=np.float32)
            output_samples = callback(input_samples, str(status) if status else None)
            outdata[:, 0] = output_samples[:frames]

        stream = sd.Stream(
            samplerate=sample_rate,
            blocksize=block_size,
            device=(self.input_device, self.output_device),
            channels=(1, 1),
            dtype="float32",
            callback=device_callback,
        )
        stream.start()
        self._stream = stream

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is not None:
            stream.stop()  # type: ignore[attr-defined]
            stream.close()  # type: ignore[attr-defined]


class RealtimeAudioStream:
    """Move samples between a duplex callback and non-realtime workers."""

    def __init__(
        self,
        backend: DuplexAudioBackend,
        *,
        sample_rate: int = 16_000,
        block_size: int = 320,
        buffer_duration_seconds: float = 1.0,
    ) -> None:
        if sample_rate < 1 or block_size < 1 or buffer_duration_seconds <= 0:
            raise ValueError("stream configuration values must be positive")
        capacity = max(block_size, round(sample_rate * buffer_duration_seconds))
        self.backend = backend
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.input_buffer = AudioRingBuffer(capacity)
        self.output_buffer = AudioRingBuffer(capacity)
        self._running = False
        self._statistics_lock = Lock()
        self._input_overflow_samples = 0
        self._output_overflow_samples = 0
        self._output_underflow_samples = 0
        self._callback_statuses: deque[str] = deque(maxlen=32)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def statistics(self) -> StreamStatistics:
        with self._statistics_lock:
            return StreamStatistics(
                self._input_overflow_samples,
                self._output_overflow_samples,
                self._output_underflow_samples,
                tuple(self._callback_statuses),
            )

    def start(self) -> None:
        if self._running:
            raise RuntimeError("audio stream is already running")
        self.backend.start(self.sample_rate, self.block_size, self._audio_callback)
        self._running = True

    def stop(self) -> None:
        if self._running:
            self.backend.stop()
            self._running = False

    def read_input(self, sample_count: int) -> AudioSamples:
        """Read captured microphone samples from a worker thread."""
        return self.input_buffer.read(sample_count)

    def write_output(self, samples: AudioSamples) -> int:
        """Queue rendered output, returning samples dropped to bound latency."""
        dropped = self.output_buffer.write(samples)
        if dropped:
            with self._statistics_lock:
                self._output_overflow_samples += dropped
        return dropped

    def _audio_callback(self, input_samples: AudioSamples, status: str | None) -> AudioSamples:
        input_dropped = self.input_buffer.write(input_samples)
        output, output_missing = self.output_buffer.read_padded(len(input_samples))
        if input_dropped or output_missing or status:
            with self._statistics_lock:
                self._input_overflow_samples += input_dropped
                self._output_underflow_samples += output_missing
                if status:
                    self._callback_statuses.append(status)
        return output


class FakeDuplexAudioBackend:
    """Deterministic callback backend for CI and offline live-engine tests."""

    def __init__(self) -> None:
        self.sample_rate: int | None = None
        self.block_size: int | None = None
        self._callback: DuplexCallback | None = None
        self._captured: list[AudioSamples] = []

    @property
    def is_running(self) -> bool:
        return self._callback is not None

    def start(
        self, sample_rate: int, block_size: int, callback: DuplexCallback
    ) -> None:
        if self.is_running:
            raise RuntimeError("fake audio backend is already running")
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._callback = callback

    def stop(self) -> None:
        self._callback = None

    def feed(self, samples: AudioSamples, *, status: str | None = None) -> AudioSamples:
        """Feed microphone audio in configured callback-sized blocks."""
        callback = self._callback
        if callback is None or self.block_size is None:
            raise RuntimeError("fake audio backend is not running")
        source = np.asarray(samples, dtype=np.float32).reshape(-1)
        outputs: list[AudioSamples] = []
        for start in range(0, len(source), self.block_size):
            output = callback(source[start : start + self.block_size], status)
            copied = np.asarray(output, dtype=np.float32).copy()
            self._captured.append(copied)
            outputs.append(copied)
        if not outputs:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(outputs).astype(np.float32, copy=False)

    def take_captured_output(self) -> AudioSamples:
        """Return captured speaker audio and clear the fake device capture."""
        if not self._captured:
            return np.empty(0, dtype=np.float32)
        result = np.concatenate(self._captured).astype(np.float32, copy=False)
        self._captured.clear()
        return result

