"""Overlap-add utilities for voicebank section transitions."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

AudioSamples = npt.NDArray[np.float32]


class BufferedOverlapComposer:
    """Mix a new section into a mutable tail before releasing audio.

    ``overlap_samples`` describes how many samples from the beginning of the
    new section should crossfade with the end of the pending section.  The
    staging window keeps recent audio mutable, which lets the live renderer
    apply overlap retroactively without changing audio already handed to the
    transport.
    """

    def __init__(self, sample_rate: int, *, staging_seconds: float = 0.0) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not math.isfinite(staging_seconds) or staging_seconds < 0:
            raise ValueError("staging_seconds must be finite and non-negative")
        self.sample_rate = sample_rate
        self.staging_samples = round(staging_seconds * sample_rate)
        self._pending = np.empty(0, dtype=np.float32)

    @property
    def pending_samples(self) -> int:
        """Number of samples currently held back for possible retroactive mixing."""
        return len(self._pending)

    def reset(self) -> None:
        """Discard pending audio."""
        self._pending = np.empty(0, dtype=np.float32)

    def append(self, samples: AudioSamples, *, overlap_samples: int = 0) -> AudioSamples:
        """Append audio and return the prefix that is now immutable."""
        incoming = np.asarray(samples, dtype=np.float32)
        if incoming.ndim != 1:
            raise ValueError("samples must be a one-dimensional array")
        overlap = max(0, int(overlap_samples))
        if overlap and len(self._pending) and len(incoming):
            count = min(overlap, len(self._pending), len(incoming))
            fade = np.linspace(0.0, 1.0, count, dtype=np.float32)
            tail = self._pending[-count:]
            self._pending[-count:] = tail * (1.0 - fade) + incoming[:count] * fade
            incoming = incoming[count:]
        if len(incoming):
            self._pending = np.concatenate((self._pending, incoming))
        return self._release_immutable()

    def flush(self) -> AudioSamples:
        """Release all pending audio and reset the staging buffer."""
        released = self._pending
        self.reset()
        return released

    def _release_immutable(self) -> AudioSamples:
        release_count = max(0, len(self._pending) - self.staging_samples)
        if release_count == 0:
            return np.empty(0, dtype=np.float32)
        released = self._pending[:release_count]
        self._pending = self._pending[release_count:]
        return released
