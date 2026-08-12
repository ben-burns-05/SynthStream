"""Offline rendering of a recognized voicebank-section timeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from synthstream.analysis import bounded_pitch_ratio, estimate_quantized_pitch_hz
from synthstream.offline.recognizer import RecognitionTimeline, TimelineSegment
from synthstream.rendering import (
    AliasEvent,
    RenderSegment,
    VoicebankRenderer,
    VoicebankRenderScheduler,
    allocate_alias_section_durations,
    fit_audio_length,
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
    track_pitch: bool = True,
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
    input_audio = _load_timeline_audio(timeline) if track_pitch else None
    total_samples = max(1, round(timeline.input_duration_seconds * output_sample_rate))
    scheduler = VoicebankRenderScheduler(
        bank,
        renderer,
        output_sample_rate,
        staging_seconds=timeline.input_duration_seconds + 1.0,
    )
    voiced_count = 0
    silence_count = 0
    overlap_count = 0

    segments = _rebalance_timeline_sections(timeline.segments, units, output_sample_rate)
    for segment in segments:
        segment_pitch_ratio = segment.pitch_ratio
        unit = units.get(segment.unit_id) if segment.unit_id is not None else None
        if input_audio is not None and unit is not None and segment.section_index is not None:
            input_start = min(
                len(input_audio), max(0, round(segment.start_seconds * timeline.sample_rate))
            )
            input_end = min(
                len(input_audio),
                max(input_start, round(segment.end_seconds * timeline.sample_rate)),
            )
            target_f0 = estimate_quantized_pitch_hz(
                input_audio[input_start:input_end],
                timeline.sample_rate,
            )
            source_f0 = renderer.estimate_section_pitch_hz(
                unit, unit.sections[segment.section_index]
            )
            segment_pitch_ratio *= bounded_pitch_ratio(target_f0, source_f0)
        result = scheduler.append(
            RenderSegment(
                unit_id=segment.unit_id,
                alias=segment.alias,
                section_index=segment.section_index,
                section_kind=segment.section_kind,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                stretch_ratio=segment.stretch_ratio,
                pitch_ratio=pitch_ratio * segment_pitch_ratio,
            )
        )
        if not result.rendered and result.target_samples == 0:
            continue
        if segment.silence:
            silence_count += 1
        else:
            voiced_count += 1
            overlap_count += int(result.overlap_samples > 0)

    if scheduler.scheduled_samples < total_samples:
        scheduler.append(
            RenderSegment(
                unit_id=None,
                alias=None,
                section_index=None,
                section_kind="silence",
                start_seconds=scheduler.scheduled_samples / output_sample_rate,
                end_seconds=total_samples / output_sample_rate,
            )
        )
    output = fit_audio_length(scheduler.flush(), total_samples)

    return VoicebankSynthesisResult(
        output,
        output_sample_rate,
        timeline,
        voiced_count,
        silence_count,
        overlap_count,
    )


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
        event = AliasEvent(
            unit_id=unit.id,
            alias=segment.alias or unit.alias,
            start_seconds=group[0].start_seconds,
            end_seconds=group[-1].end_seconds,
            pitch_ratio=group[0].pitch_ratio,
        )
        durations = allocate_alias_section_durations(
            event,
            unit,
            timebase_hz=sample_rate,
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


def _load_timeline_audio(timeline: RecognitionTimeline) -> AudioSamples | None:
    """Load and normalize the source audio used for per-section pitch targets."""
    try:
        waveform, source_rate = sf.read(
            timeline.source_wav, dtype="float32", always_2d=True
        )
    except (OSError, RuntimeError, sf.LibsndfileError):
        return None
    if not len(waveform) or source_rate < 1:
        return None
    mono = np.asarray(np.mean(waveform, axis=1), dtype=np.float32)
    if source_rate == timeline.sample_rate:
        return mono
    divisor = math.gcd(source_rate, timeline.sample_rate)
    return np.asarray(
        resample_poly(
            mono,
            timeline.sample_rate // divisor,
            source_rate // divisor,
        ),
        dtype=np.float32,
    )
