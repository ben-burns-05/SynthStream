"""Offline rendering of a recognized voicebank-section timeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from synthstream.offline.recognizer import RecognitionTimeline
from synthstream.rendering import VoicebankRenderer
from synthstream.voicebank import Voicebank

AudioSamples = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class VoicebankSynthesisResult:
    """Rendered waveform and provenance for one recognized timeline."""

    samples: AudioSamples
    sample_rate: int
    source_timeline: RecognitionTimeline
    voiced_segments: int
    silence_segments: int

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
    output = np.zeros(total_samples, dtype=np.float32)
    voiced_count = 0
    silence_count = 0

    for segment in timeline.segments:
        start = min(total_samples, max(0, round(segment.start_seconds * output_sample_rate)))
        end = min(total_samples, max(start, round(segment.end_seconds * output_sample_rate)))
        if end <= start:
            continue
        if segment.silence:
            silence_count += 1
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
        rendered = renderer.render_section(
            unit,
            unit.sections[segment.section_index],
            duration_seconds=target_samples / output_sample_rate,
            pitch_ratio=pitch_ratio,
        )
        converted = _resample(rendered.samples, rendered.sample_rate, output_sample_rate)
        output[start:end] = _fit_length(converted, target_samples)
        voiced_count += 1

    return VoicebankSynthesisResult(
        output,
        output_sample_rate,
        timeline,
        voiced_count,
        silence_count,
    )


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
