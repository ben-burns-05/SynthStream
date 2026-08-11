"""Offline rendering of a recognized voicebank-section timeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from synthstream.offline.recognizer import RecognitionTimeline, TimelineSegment
from synthstream.rendering import (
    BufferedOverlapComposer,
    VoicebankRenderer,
    rebalance_section_durations,
)
from synthstream.voicebank import Voicebank, VoicebankUnit

AudioSamples = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class VoicebankSynthesisResult:
    """Rendered waveform and provenance for one recognized timeline."""

    samples: AudioSamples
    sample_rate: int
    source_timeline: RecognitionTimeline
    voiced_segments: int
    silence_segments: int
    overlap_segments: int = 0

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    def write_wav(self, path: str | Path, *, subtype: str = "PCM_16") -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, self.samples, self.sample_rate, subtype=subtype, format="WAV")
        return output_path


def synthesize_timeline(
    timeline: RecognitionTimeline,
    bank: Voicebank,
    *,
    output_sample_rate: int | None = None,
    pitch_ratio: float = 1.0,
    output_gain: float = 1.0,
) -> VoicebankSynthesisResult:
    """Render every selected OTO section into one contiguous voicebank WAV."""
    if output_sample_rate is None:
        output_sample_rate = _default_sample_rate(bank)
    if output_sample_rate <= 0:
        raise ValueError("output_sample_rate must be positive")
    if not math.isfinite(pitch_ratio) or pitch_ratio <= 0:
        raise ValueError("pitch_ratio must be finite and positive")

    units = {unit.id: unit for unit in bank.units}
    renderer = VoicebankRenderer(output_gain=output_gain)
    total_samples = max(1, round(timeline.input_duration_seconds * output_sample_rate))
    # Offline output can keep the whole timeline mutable until the final WAV
    # is assembled; the live engine uses a short staging window instead.
    composer = BufferedOverlapComposer(
        output_sample_rate,
        staging_seconds=timeline.input_duration_seconds + 1.0,
    )
    voiced_count = 0
    silence_count = 0
    overlap_count = 0
    scheduled_samples = 0
    previous_unit_id: str | None = None
    onset_stretch_by_unit: dict[str, float] = {}

    segments = _rebalance_timeline_sections(timeline.segments, units, output_sample_rate)
    for segment in segments:
        start = min(total_samples, max(0, round(segment.start_seconds * output_sample_rate)))
        end = min(total_samples, max(start, round(segment.end_seconds * output_sample_rate)))
        if end <= start:
            continue
        if start > scheduled_samples:
            composer.append(
                np.zeros(start - scheduled_samples, dtype=np.float32), overlap_samples=0
            )
            previous_unit_id = None
        scheduled_samples = start
        if segment.silence:
            silence_count += 1
            composer.append(np.zeros(end - start, dtype=np.float32), overlap_samples=0)
            scheduled_samples = end
            previous_unit_id = None
            continue
        if segment.unit_id is None or segment.section_index is None:
            raise ValueError("voiced timeline segment is missing unit or section identity")
        unit = units.get(segment.unit_id)
        if unit is None:
            raise ValueError(f"timeline references unknown voicebank unit: {segment.unit_id}")
        if not 0 <= segment.section_index < len(unit.sections):
            raise ValueError(
                f"timeline references invalid section {segment.section_index} for {unit.id}"
            )
        target_samples = end - start
        overlap_samples = _overlap_samples(
            unit,
            segment,
            previous_unit_id,
            onset_stretch_by_unit,
            output_sample_rate,
            target_samples,
        )
        if segment.section_kind == "onset":
            onset_stretch_by_unit[unit.id] = max(segment.stretch_ratio, 1e-6)
        rendered = renderer.render_section(
            unit,
            unit.sections[segment.section_index],
            duration_seconds=(target_samples + overlap_samples) / output_sample_rate,
            pitch_ratio=pitch_ratio,
        )
        converted = _resample(rendered.samples, rendered.sample_rate, output_sample_rate)
        composer.append(
            _fit_length(converted, target_samples + overlap_samples),
            overlap_samples=overlap_samples,
        )
        voiced_count += 1
        overlap_count += int(overlap_samples > 0)
        scheduled_samples = end
        previous_unit_id = unit.id

    if scheduled_samples < total_samples:
        composer.append(
            np.zeros(total_samples - scheduled_samples, dtype=np.float32), overlap_samples=0
        )
    output = _fit_length(composer.flush(), total_samples)

    return VoicebankSynthesisResult(
        output,
        output_sample_rate,
        timeline,
        voiced_count,
        silence_count,
        overlap_count,
    )


def _overlap_samples(
    unit: VoicebankUnit,
    segment: TimelineSegment,
    previous_unit_id: str | None,
    onset_stretch_by_unit: dict[str, float],
    sample_rate: int,
    target_samples: int,
) -> int:
    """Return warped OTO overlap for a transition into a new alias unit."""
    unit_id = unit.id
    if (
        previous_unit_id == unit_id
        and segment.section_index is not None
        and segment.section_index > 0
    ):
        return min(target_samples, round(0.005 * sample_rate))
    if previous_unit_id is None or previous_unit_id == unit_id:
        return 0
    overlap_ms = max(0.0, unit.overlap_ms)
    if overlap_ms <= 0:
        return 0
    stretch = onset_stretch_by_unit.get(unit_id, 1.0)
    if segment.section_kind == "onset":
        stretch = max(segment.stretch_ratio, 1e-6)
    return min(target_samples, max(0, round(overlap_ms * sample_rate / 1000.0 * stretch)))


def _rebalance_timeline_sections(
    segments: tuple[TimelineSegment, ...],
    units: dict[str, VoicebankUnit],
    sample_rate: int,
) -> tuple[TimelineSegment, ...]:
    """Reallocate alias timing so transients stay close to source duration."""
    result: list[TimelineSegment] = []
    index = 0
    while index < len(segments):
        segment = segments[index]
        if segment.silence or segment.unit_id is None or segment.section_index != 0:
            result.append(segment)
            index += 1
            continue
        unit = units.get(segment.unit_id)
        if unit is None:
            result.append(segment)
            index += 1
            continue
        group = [segment]
        cursor = index + 1
        while cursor < len(segments):
            candidate = segments[cursor]
            if (
                candidate.silence
                or candidate.unit_id != segment.unit_id
                or candidate.alias != segment.alias
                or candidate.section_index != len(group)
            ):
                break
            group.append(candidate)
            cursor += 1
        if len(group) != len(unit.sections):
            result.extend(group)
            index = cursor
            continue
        start_sample = round(group[0].start_seconds * sample_rate)
        target_samples = tuple(
            max(1, round((item.end_seconds - item.start_seconds) * sample_rate))
            for item in group
        )
        durations = rebalance_section_durations(
            unit.sections,
            target_samples,
            sample_rate=sample_rate,
        )
        current = start_sample
        for item, duration, section in zip(group, durations, unit.sections, strict=True):
            next_sample = current + duration
            result.append(
                replace(
                    item,
                    start_seconds=current / sample_rate,
                    end_seconds=next_sample / sample_rate,
                    duration_seconds=duration / sample_rate,
                    stretch_ratio=(duration / sample_rate) / section.duration_seconds,
                )
            )
            current = next_sample
        index = cursor
    return tuple(result)


def _default_sample_rate(bank: Voicebank) -> int:
    if not bank.units:
        raise ValueError("cannot synthesize an empty voicebank")
    return bank.units[0].sample_rate


def _resample(samples: AudioSamples, source_rate: int, target_rate: int) -> AudioSamples:
    if source_rate == target_rate:
        return samples
    divisor = math.gcd(source_rate, target_rate)
    return np.asarray(
        resample_poly(samples, target_rate // divisor, source_rate // divisor),
        dtype=np.float32,
    )


def _fit_length(samples: AudioSamples, target_samples: int) -> AudioSamples:
    if len(samples) == target_samples:
        return samples
    if len(samples) > target_samples:
        return np.asarray(samples[:target_samples], dtype=np.float32)
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: len(samples)] = samples
    return padded
