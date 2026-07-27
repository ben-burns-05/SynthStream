"""Bounded thread-safe audio sample buffering."""

from threading import Lock

import numpy as np

from synthstream.audio.output import AudioSamples


class AudioRingBuffer:
    """A fixed-capacity mono float32 ring buffer.

    New audio wins on overflow so queued latency cannot grow without bound.
    """

    def __init__(self, capacity_samples: int) -> None:
        if capacity_samples < 1:
            raise ValueError("capacity_samples must be positive")
        self._samples = np.zeros(capacity_samples, dtype=np.float32)
        self._read_index = 0
        self._size = 0
        self._lock = Lock()

    @property
    def capacity_samples(self) -> int:
        return len(self._samples)

    @property
    def available_samples(self) -> int:
        with self._lock:
            return self._size

    def clear(self) -> None:
        with self._lock:
            self._read_index = 0
            self._size = 0

    def write(self, samples: AudioSamples) -> int:
        """Append samples and return the number discarded by overflow."""
        incoming = np.asarray(samples, dtype=np.float32).reshape(-1)
        with self._lock:
            dropped = max(0, self._size + len(incoming) - self.capacity_samples)
            if len(incoming) >= self.capacity_samples:
                dropped = self._size + len(incoming) - self.capacity_samples
                incoming = incoming[-self.capacity_samples :]
                self._read_index = 0
                self._size = 0
            elif dropped:
                self._read_index = (self._read_index + dropped) % self.capacity_samples
                self._size -= dropped

            write_index = (self._read_index + self._size) % self.capacity_samples
            first_count = min(len(incoming), self.capacity_samples - write_index)
            self._samples[write_index : write_index + first_count] = incoming[:first_count]
            remaining = len(incoming) - first_count
            if remaining:
                self._samples[:remaining] = incoming[first_count:]
            self._size += len(incoming)
            return dropped

    def read(self, sample_count: int) -> AudioSamples:
        """Remove and return up to ``sample_count`` available samples."""
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        with self._lock:
            count = min(sample_count, self._size)
            result = np.empty(count, dtype=np.float32)
            first_count = min(count, self.capacity_samples - self._read_index)
            result[:first_count] = self._samples[
                self._read_index : self._read_index + first_count
            ]
            remaining = count - first_count
            if remaining:
                result[first_count:] = self._samples[:remaining]
            self._read_index = (self._read_index + count) % self.capacity_samples
            self._size -= count
            return result

    def read_padded(self, sample_count: int) -> tuple[AudioSamples, int]:
        """Read exactly ``sample_count`` samples, padding underflow with silence."""
        result = self.read(sample_count)
        missing = sample_count - len(result)
        if missing:
            result = np.pad(result, (0, missing)).astype(np.float32, copy=False)
        return result, missing

