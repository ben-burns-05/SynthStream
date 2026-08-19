"""Shared timeline scheduling for live and offline voicebank rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from synthstream.rendering.overlap import BufferedOverlapComposer
from synthstream.rendering.renderer import VoicebankRenderer
from synthstream.voicebank import Voicebank, VoicebankUnit

AudioSamples = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class RenderSegment:
    """One timeline item presented to the shared render scheduler."""

    unit_id: str | None
    alias: str | None
    section_index: int | None
    section_kind: str
    start_seconds: float
    end_seconds: float
    pitch_ratio: float = 1.0

    @property
    def is_silence(self) -> bool:
        return self.unit_id is None


@dataclass(frozen=True, slots=True)
class RenderAppend:
    """Result of scheduling one segment."""

    released: AudioSamples
    target_samples: int
    overlap_samples: int
    rendered: bool


class VoicebankRenderScheduler:
    """Place, render, and overlap-add timeline segments consistently.

    The scheduler owns the mutable overlap staging buffer.  Callers only need
    to convert their decoder output into :class:`RenderSegment` values and
    consume ``released`` samples (or collect them for an offline render).
    """

    def __init__(
        self,
        bank: Voicebank,
        renderer: VoicebankRenderer,
        output_sample_rate: int,
        *,
        staging_seconds: float = 0.0,
    ) -> None:
        if output_sample_rate <= 0:
            raise ValueError("output_sample_rate must be positive")
        self.units = {unit.id: unit for unit in bank.units}
        self.renderer = renderer
        self.output_sample_rate = output_sample_rate
        self._composer = BufferedOverlapComposer(
            output_sample_rate,
            staging_seconds=staging_seconds,
        )
        self._scheduled_samples = 0
        self._previous_unit_id: str | None = None

    @property
    def scheduled_samples(self) -> int:
        """End of the latest scheduled timeline item in output samples."""
        return self._scheduled_samples

    def reset(self) -> None:
        """Reset timeline state and discard staged audio."""
        self._composer.reset()
        self._scheduled_samples = 0
        self._previous_unit_id = None

    def append(
        self,
        segment: RenderSegment,
        *,
        include_leading_gap: bool = True,
        leading_gap_limit_samples: int | None = None,
    ) -> RenderAppend:
        """Schedule one silence or voiced section and return immutable output.

        Offline rendering includes timeline gaps as explicit silence. Live
        callers may anchor the first event and bound later leading gaps with
        ``leading_gap_limit_samples``; the physical output clock can then
        cover older silence while the worker catches up.
        """
        start_sample = max(0, round(segment.start_seconds * self.output_sample_rate))
        end_sample = max(start_sample, round(segment.end_seconds * self.output_sample_rate))
        if end_sample <= start_sample:
            return RenderAppend(_empty_audio(), 0, 0, False)

        released = _empty_audio()
        if leading_gap_limit_samples is not None and leading_gap_limit_samples < 0:
            raise ValueError("leading_gap_limit_samples must be non-negative")
        gap_samples = max(0, start_sample - self._scheduled_samples)
        if include_leading_gap and gap_samples:
            if leading_gap_limit_samples is not None:
                gap_samples = min(gap_samples, leading_gap_limit_samples)
                # A bounded live queue cannot retain an arbitrarily old
                # silence interval.  Keep the most recent part of the gap;
                # the device has already emitted the discarded prefix.
                self._scheduled_samples = start_sample - gap_samples
            released = self._composer.append(
                np.zeros(gap_samples, dtype=np.float32),
                overlap_samples=0,
            )
            self._scheduled_samples = start_sample
            self._previous_unit_id = None
        elif start_sample > self._scheduled_samples:
            self._scheduled_samples = start_sample
            self._previous_unit_id = None

        target_samples = end_sample - start_sample
        if segment.is_silence:
            rendered = np.zeros(target_samples, dtype=np.float32)
            overlap_samples = 0
            self._previous_unit_id = None
        else:
            unit = self._resolve_unit(segment)
            overlap_samples = self._overlap_samples(unit, segment, target_samples)
            result = self.renderer.render_section(
                unit,
                unit.section_at(segment.section_index or 0),
                duration_seconds=(target_samples + overlap_samples) / self.output_sample_rate,
                pitch_ratio=segment.pitch_ratio,
            )
            rendered = fit_audio_length(
                resample_audio(result.samples, result.sample_rate, self.output_sample_rate),
                target_samples + overlap_samples,
            )
            self._previous_unit_id = unit.id

        appended = self._composer.append(rendered, overlap_samples=overlap_samples)
        released = np.concatenate((released, appended)) if len(released) else appended
        self._scheduled_samples = max(self._scheduled_samples, end_sample)
        return RenderAppend(released, target_samples, overlap_samples, not segment.is_silence)

    def flush(self) -> AudioSamples:
        """Release all staged output and reset the overlap buffer."""
        return self._composer.flush()

    def _resolve_unit(self, segment: RenderSegment) -> VoicebankUnit:
        if segment.unit_id is None or segment.section_index is None:
            raise ValueError("voiced segment is missing voicebank identity")
        unit = self.units.get(segment.unit_id)
        if unit is None:
            raise ValueError("segment references an unavailable voicebank section")
        try:
            unit.section_at(segment.section_index)
        except IndexError as error:
            raise ValueError("segment references an unavailable voicebank section") from error
        return unit

    def _overlap_samples(
        self,
        unit: VoicebankUnit,
        segment: RenderSegment,
        target_samples: int,
    ) -> int:
        # Sections within one alias remain contiguous. OTO overlap belongs to
        # the onset of every subsequent alias event, even when it reuses the
        # same recording unit.
        if segment.section_index != 0 or self._previous_unit_id is None:
            return 0
        overlap_ms = max(0.0, unit.overlap_ms)
        if overlap_ms <= 0:
            return 0
        return min(
            target_samples,
            max(0, round(overlap_ms * self.output_sample_rate / 1000.0)),
        )


def resample_audio(samples: AudioSamples, source_rate: int, target_rate: int) -> AudioSamples:
    """Resample mono float audio when source and output rates differ."""
    if source_rate == target_rate:
        return np.asarray(samples, dtype=np.float32)
    divisor = math.gcd(source_rate, target_rate)
    return np.asarray(
        resample_poly(samples, target_rate // divisor, source_rate // divisor),
        dtype=np.float32,
    )


def fit_audio_length(samples: AudioSamples, target_samples: int) -> AudioSamples:
    """Trim or zero-pad audio to an exact sample count."""
    if len(samples) == target_samples:
        return np.asarray(samples, dtype=np.float32)
    if len(samples) > target_samples:
        return np.asarray(samples[:target_samples], dtype=np.float32)
    result = np.zeros(target_samples, dtype=np.float32)
    result[: len(samples)] = samples
    return result


def _empty_audio() -> AudioSamples:
    return np.empty(0, dtype=np.float32)
