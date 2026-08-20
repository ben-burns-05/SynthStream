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
from synthstream.rendering.events import allocate_sustain_only_durations
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
        """Render one complete alias event from the usable unit region."""
        return self.render_alias(
            unit,
            duration_seconds=duration_seconds,
            pitch_ratio=pitch_ratio,
        )

    def render_alias(
        self,
        unit: VoicebankUnit,
        *,
        duration_seconds: float,
        pitch_ratio: float = 1.0,
    ) -> RenderResult:
        """Render an alias as one pitch operation with sustain-only stretching.

        The source sections are assembled before pitch shifting.  Onset and
        transition samples retain their recorded lengths whenever the event is
        long enough; any remaining duration is assigned to sustain.  This
        avoids resetting the phase vocoder at every internal section boundary.
        """
        _validate_transform(duration_seconds, pitch_ratio)
        source = _read_sections(unit)
        target_samples = round(duration_seconds * unit.sample_rate)
        if target_samples < 1:
            raise ValueError("duration_seconds is shorter than one output sample")
        section_targets = allocate_sustain_only_durations(
            unit.sections,
            target_samples,
            timebase_hz=unit.sample_rate,
        )
        pitched = _pitch_shift(source, unit.sample_rate, pitch_ratio)
        rendered_sections: list[AudioSamples] = []
        source_cursor = 0
        for section, section_target in zip(
            unit.sections, section_targets, strict=True
        ):
            source_end = source_cursor + section.sample_count
            section_audio = pitched[source_cursor:source_end]
            if section_target != len(section_audio):
                # In the normal path only sustain differs from its recorded
                # duration.  The fallback is needed for very short events
                # where even protected transients cannot fit.
                section_audio = _time_stretch(section_audio, section_target)
            rendered_sections.append(section_audio)
            source_cursor = source_end
        transformed = _fit_audio_length(
            np.concatenate(rendered_sections).astype(np.float32, copy=False),
            target_samples,
        )
        transformed = np.nan_to_num(transformed * self.output_gain, copy=False)
        transformed = np.clip(transformed, -1.0, 1.0).astype(np.float32, copy=False)
        return RenderResult(
            samples=transformed,
            sample_rate=unit.sample_rate,
            unit_id=unit.id,
            source_duration_seconds=len(source) / unit.sample_rate,
            target_duration_seconds=target_samples / unit.sample_rate,
            stretch_ratio=target_samples / len(source),
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


def _read_sections(unit: VoicebankUnit) -> AudioSamples:
    """Read and concatenate the exact usable source sections once."""
    chunks: list[AudioSamples] = []
    for section in unit.sections:
        waveform, file_sample_rate = sf.read(
            unit.wav_path,
            start=section.start_sample,
            stop=section.end_sample,
            dtype="float32",
            always_2d=True,
        )
        if file_sample_rate != unit.sample_rate:
            raise ValueError("WAV sample rate changed after voicebank loading")
        chunks.append(np.asarray(np.mean(waveform, axis=1), dtype=np.float32))
    if not chunks:
        raise ValueError("voicebank unit has no usable sections")
    return np.concatenate(chunks).astype(np.float32, copy=False)


def _fit_audio_length(samples: AudioSamples, target_samples: int) -> AudioSamples:
    if len(samples) == target_samples:
        return np.asarray(samples, dtype=np.float32)
    if len(samples) > target_samples:
        return np.asarray(samples[:target_samples], dtype=np.float32)
    result = np.zeros(target_samples, dtype=np.float32)
    result[: len(samples)] = samples
    return result


def _largest_power_of_two(value: int) -> int:
    return 1 << (value.bit_length() - 1)
