"""Canonical alias events and their internal section duration allocation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from synthstream.voicebank import VoicebankSection, VoicebankUnit


@dataclass(frozen=True, slots=True)
class AliasEvent:
    """One selected voicebank alias spanning all of its source sections."""

    unit_id: str
    alias: str
    start_seconds: float
    end_seconds: float
    confidence: float = 1.0
    pitch_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not self.unit_id or not self.alias:
            raise ValueError("alias events require unit and alias identities")
        if (
            not math.isfinite(self.start_seconds)
            or not math.isfinite(self.end_seconds)
            or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("alias event timing must be finite and increasing")
        if not math.isfinite(self.confidence) or self.confidence < 0:
            raise ValueError("alias event confidence must be finite and non-negative")
        if not math.isfinite(self.pitch_ratio) or self.pitch_ratio <= 0:
            raise ValueError("alias event pitch ratio must be finite and positive")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def allocate_alias_section_durations(
    event: AliasEvent,
    unit: VoicebankUnit,
    *,
    timebase_hz: float,
) -> tuple[int, ...]:
    """Expand one alias event while reserving onset and transition durations.

    ``timebase_hz`` is the output timing grid: the live hop rate when creating
    decoder frames, or the output sample rate during offline assembly. Sustain
    receives all duration beyond the recorded transient sections. If an event
    is shorter than the protected transient material, a proportional fallback
    is used because no positive-duration schedule can otherwise be represented.
    """
    if event.unit_id != unit.id:
        raise ValueError("alias event does not reference the supplied unit")
    if not math.isfinite(timebase_hz) or timebase_hz <= 0:
        raise ValueError("timebase_hz must be finite and positive")
    if not unit.sections:
        raise ValueError("voicebank unit has no sections")

    return allocate_sustain_only_durations(
        unit.sections,
        round(event.duration_seconds * timebase_hz),
        timebase_hz=timebase_hz,
    )


def allocate_sustain_only_durations(
    sections: tuple[VoicebankSection, ...],
    total_units: int,
    *,
    timebase_hz: float,
) -> tuple[int, ...]:
    """Allocate a duration grid while reserving onset and transition units."""
    if not sections:
        raise ValueError("voicebank unit has no sections")
    if not math.isfinite(timebase_hz) or timebase_hz <= 0:
        raise ValueError("timebase_hz must be finite and positive")
    total = max(len(sections), int(total_units))
    nominal = tuple(
        max(1, round(section.duration_seconds * timebase_hz))
        for section in sections
    )
    transient_indices = tuple(
        index
        for index, section in enumerate(sections)
        if section.kind in {"onset", "transition"}
    )
    sustain_indices = tuple(
        index for index, section in enumerate(sections) if section.kind == "sustain"
    )
    if not sustain_indices:
        return _proportional_durations(total, nominal)

    protected = [0] * len(sections)
    transient_total = sum(nominal[index] for index in transient_indices)
    minimum_sustain = len(sustain_indices)
    if transient_total + minimum_sustain <= total:
        for index in transient_indices:
            protected[index] = nominal[index]
        remaining = total - transient_total
        sustain_durations = _proportional_durations(
            remaining, tuple(nominal[index] for index in sustain_indices)
        )
        for index, duration in zip(sustain_indices, sustain_durations, strict=True):
            protected[index] = duration
        return tuple(protected)

    # A very short alias cannot fit both the recorded transients and a positive
    # sustain. Preserve the total duration with the least surprising fallback.
    transient_budget = max(len(transient_indices), total - minimum_sustain)
    transient_durations = _proportional_durations(
        transient_budget, tuple(nominal[index] for index in transient_indices)
    )
    for index, duration in zip(transient_indices, transient_durations, strict=True):
        protected[index] = duration
    sustain_durations = _proportional_durations(
        total - sum(transient_durations),
        tuple(nominal[index] for index in sustain_indices),
    )
    for index, duration in zip(sustain_indices, sustain_durations, strict=True):
        protected[index] = duration
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
