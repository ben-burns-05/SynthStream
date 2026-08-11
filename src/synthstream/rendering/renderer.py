"""Render selected voicebank recordings at requested durations and pitches."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch
import torchaudio.functional as audio_functional  # type: ignore[import-untyped]
from scipy.signal import resample  # type: ignore[import-untyped]

from synthstream.audio.output import AudioSamples, AudioSink
from synthstream.voicebank.models import VoicebankSection, VoicebankUnit


@dataclass(frozen=True, slots=True)
class RenderResult:
    """A transformed waveform and the parameters that produced it."""

    samples: AudioSamples
    sample_rate: int
    unit_id: str
    source_duration_seconds: float
    target_duration_seconds: float
    stretch_ratio: float
    pitch_ratio: float

    def send_to(self, sink: AudioSink) -> None:
        """Send this result to a file, device, or test sink."""
        sink.write(self.samples, self.sample_rate)


class VoicebankRenderer:
    """Transform original voicebank regions without changing recognition models."""

    def __init__(self, *, output_gain: float = 1.0) -> None:
        if not math.isfinite(output_gain) or output_gain < 0:
            raise ValueError("output_gain must be finite and non-negative")
        self.output_gain = output_gain

    def render_unit(
        self,
        unit: VoicebankUnit,
        *,
        duration_seconds: float,
        pitch_ratio: float = 1.0,
    ) -> RenderResult:
        """Render the complete usable region of one voicebank unit."""
        return self._render(
            unit,
            unit.sections,
            duration_seconds=duration_seconds,
            pitch_ratio=pitch_ratio,
        )

    def render_section(
        self,
        unit: VoicebankUnit,
        section: VoicebankSection,
        *,
        duration_seconds: float,
        pitch_ratio: float = 1.0,
    ) -> RenderResult:
        """Render one section belonging to a voicebank unit."""
        if section not in unit.sections:
            raise ValueError("section does not belong to the supplied unit")
        return self._render(
            unit,
            (section,),
            duration_seconds=duration_seconds,
            pitch_ratio=pitch_ratio,
        )

    def _render(
        self,
        unit: VoicebankUnit,
        sections: tuple[VoicebankSection, ...],
        *,
        duration_seconds: float,
        pitch_ratio: float,
    ) -> RenderResult:
        _validate_transform(duration_seconds, pitch_ratio)
        first, last = sections[0], sections[-1]
        waveform, file_sample_rate = sf.read(
            unit.wav_path,
            start=first.start_sample,
            stop=last.end_sample,
            dtype="float32",
            always_2d=True,
        )
        if file_sample_rate != unit.sample_rate:
            raise ValueError("WAV sample rate changed after voicebank loading")
        mono: AudioSamples = np.asarray(np.mean(waveform, axis=1), dtype=np.float32)
        target_samples = round(duration_seconds * unit.sample_rate)
        if target_samples < 1:
            raise ValueError("duration_seconds is shorter than one output sample")

        transformed = _pitch_shift(mono, unit.sample_rate, pitch_ratio)
        transformed = _time_stretch(transformed, target_samples)
        transformed = np.nan_to_num(transformed * self.output_gain, copy=False)
        transformed = np.clip(transformed, -1.0, 1.0).astype(np.float32, copy=False)
        source_duration = len(mono) / unit.sample_rate
        return RenderResult(
            samples=transformed,
            sample_rate=unit.sample_rate,
            unit_id=unit.id,
            source_duration_seconds=source_duration,
            target_duration_seconds=target_samples / unit.sample_rate,
            stretch_ratio=target_samples / len(mono),
            pitch_ratio=pitch_ratio,
        )


def _validate_transform(duration_seconds: float, pitch_ratio: float) -> None:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive")
    if not math.isfinite(pitch_ratio) or pitch_ratio <= 0:
        raise ValueError("pitch_ratio must be finite and positive")


def _pitch_shift(samples: AudioSamples, sample_rate: int, pitch_ratio: float) -> AudioSamples:
    del sample_rate
    if math.isclose(pitch_ratio, 1.0):
        return samples.copy()
    stretched_length = max(1, round(len(samples) * pitch_ratio))
    stretched = _time_stretch(samples, stretched_length)
    return np.asarray(resample(stretched, len(samples)), dtype=np.float32)


def _time_stretch(samples: AudioSamples, target_samples: int) -> AudioSamples:
    if len(samples) == target_samples:
        return samples.copy()

    n_fft = min(1024, _largest_power_of_two(max(32, len(samples))))
    hop_length = max(1, n_fft // 4)
    window = torch.hann_window(n_fft)
    waveform = torch.from_numpy(samples)
    spectrum = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        pad_mode="constant",
        return_complex=True,
    )
    rate = len(samples) / target_samples
    phase_advance = torch.linspace(0, math.pi * hop_length, spectrum.shape[-2]).unsqueeze(-1)
    stretched_spectrum: torch.Tensor = audio_functional.phase_vocoder(
        spectrum, rate, phase_advance
    )
    stretched = torch.istft(
        stretched_spectrum,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        length=target_samples,
    )
    return np.asarray(stretched.numpy(), dtype=np.float32)


def _largest_power_of_two(value: int) -> int:
    return 1 << (value.bit_length() - 1)


def rebalance_section_durations(
    sections: tuple[VoicebankSection, ...],
    target_samples: tuple[int, ...],
    *,
    sample_rate: int,
    transient_min_ratio: float = 0.75,
    transient_max_ratio: float = 1.25,
) -> tuple[int, ...]:
    """Keep attacks near source duration and assign excess time to sustain.

    The returned durations preserve the original total duration.  Onset and
    transition sections are clamped to a moderate stretch range; sustain
    receives the remaining samples.  This prevents short transients from
    being phase-vocoder-stretched by several hundred percent.
    """
    if len(sections) != len(target_samples):
        raise ValueError("sections and target_samples must have equal length")
    if sample_rate <= 0 or not sections:
        raise ValueError("sample_rate and sections must be positive")
    if not 0 < transient_min_ratio <= transient_max_ratio:
        raise ValueError("transient stretch bounds are invalid")
    total = max(len(sections), sum(max(1, int(value)) for value in target_samples))
    nominal = tuple(
        max(1, round(section.duration_seconds * sample_rate)) for section in sections
    )
    transients = tuple(
        index
        for index, section in enumerate(sections)
        if section.kind in {"onset", "transition"}
    )
    sustains = tuple(
        index for index, section in enumerate(sections) if section.kind == "sustain"
    )
    if not transients or not sustains:
        return _proportional_durations(total, target_samples)

    protected = [0] * len(sections)
    for index in transients:
        lower = max(1, round(nominal[index] * transient_min_ratio))
        upper = max(lower, round(nominal[index] * transient_max_ratio))
        protected[index] = min(upper, max(lower, int(target_samples[index])))

    available_for_transients = max(len(transients), total - len(sustains))
    if sum(protected) > available_for_transients:
        scaled = _proportional_durations(
            available_for_transients, tuple(protected[index] for index in transients)
        )
        for index, value in zip(transients, scaled, strict=True):
            protected[index] = value

    remaining = total - sum(protected)
    sustain_weights = tuple(max(1, int(target_samples[index])) for index in sustains)
    sustain_durations = _proportional_durations(remaining, sustain_weights)
    for index, value in zip(sustains, sustain_durations, strict=True):
        protected[index] = value
    return tuple(protected)


def _proportional_durations(total: int, weights: tuple[int, ...]) -> tuple[int, ...]:
    if not weights:
        return ()
    total = max(len(weights), int(total))
    weight_sum = max(1, sum(max(1, value) for value in weights))
    raw = [total * max(1, value) / weight_sum for value in weights]
    result = [max(1, int(value)) for value in raw]
    remainder = total - sum(result)
    order = sorted(
        range(len(weights)),
        key=lambda index: raw[index] - int(raw[index]),
        reverse=True,
    )
    while remainder > 0:
        for index in order:
            if remainder <= 0:
                break
            result[index] += 1
            remainder -= 1
    while remainder < 0:
        for index in reversed(order):
            if remainder >= 0:
                break
            if result[index] > 1:
                result[index] -= 1
                remainder += 1
    return tuple(result)
