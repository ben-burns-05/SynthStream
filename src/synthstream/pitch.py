"""Small, shared pitch-transfer policy for human audio and voicebanks."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]

from synthstream.analysis import bounded_pitch_ratio, estimate_quantized_pitch_hz
from synthstream.voicebank import VoicebankSection, VoicebankUnit

AudioSamples = npt.NDArray[np.float32]


class VoicebankPitchCache:
    """Cache one quantized recorded pitch estimate per voicebank section."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], float | None] = {}

    def estimate_section_pitch_hz(
        self, unit: VoicebankUnit, section: VoicebankSection
    ) -> float | None:
        """Estimate a section's recorded F0, validating the section ownership."""
        if section not in unit.sections:
            raise ValueError("section does not belong to the supplied unit")
        key = (unit.id, section.start_sample, section.end_sample)
        if key not in self._cache:
            waveform, file_sample_rate = sf.read(
                unit.wav_path,
                start=section.start_sample,
                stop=section.end_sample,
                dtype="float32",
                always_2d=True,
            )
            if file_sample_rate != unit.sample_rate:
                raise ValueError("WAV sample rate changed after voicebank loading")
            mono = np.asarray(np.mean(waveform, axis=1), dtype=np.float32)
            self._cache[key] = estimate_quantized_pitch_hz(mono, unit.sample_rate)
        return self._cache[key]

    def estimate_unit_pitch_hz(self, unit: VoicebankUnit) -> float | None:
        """Estimate the unit pitch from sustain, falling back to its final section."""
        return self.estimate_section_pitch_hz(unit, unit.pitch_reference_section)


class PitchTransfer:
    """Transfer one nearest-note F0 estimate onto one voicebank alias event."""

    def __init__(self, cache: VoicebankPitchCache | None = None) -> None:
        self.cache = cache or VoicebankPitchCache()

    def ratio_for_alias(
        self,
        target_audio: npt.ArrayLike,
        target_sample_rate: int,
        unit: VoicebankUnit,
    ) -> float:
        """Return a bounded source-to-target ratio, or unity when unvoiced."""
        target_f0 = estimate_quantized_pitch_hz(target_audio, target_sample_rate)
        source_f0 = self.cache.estimate_unit_pitch_hz(unit)
        return bounded_pitch_ratio(target_f0, source_f0)


__all__ = ["PitchTransfer", "VoicebankPitchCache"]
