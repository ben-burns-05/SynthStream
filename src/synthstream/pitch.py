"""Small, shared pitch-transfer policy for human audio and voicebanks."""

from __future__ import annotations

import numpy.typing as npt

from synthstream.analysis import bounded_pitch_ratio, estimate_quantized_pitch_hz
from synthstream.voicebank.models import VoicebankUnit


class PitchTransfer:
    """Transfer one nearest-note F0 estimate onto one voicebank alias event."""

    def ratio_for_alias(
        self,
        target_audio: npt.ArrayLike,
        target_sample_rate: int,
        unit: VoicebankUnit,
    ) -> float:
        """Return a bounded source-to-target ratio, or unity when unvoiced."""
        target_f0 = estimate_quantized_pitch_hz(target_audio, target_sample_rate)
        source_f0 = unit.source_pitch_hz
        return bounded_pitch_ratio(target_f0, source_f0)


__all__ = ["PitchTransfer"]
