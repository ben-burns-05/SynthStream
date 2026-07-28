"""Generic voicebank models shared by matching and rendering."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VoicebankSection:
    """A contiguous subsection of a voicebank recording."""

    kind: str
    start_sample: int
    end_sample: int
    sample_rate: int

    @property
    def sample_count(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate


@dataclass(frozen=True, slots=True)
class VoicebankUnit:
    """One usable ``oto.ini`` entry and its source/rendering metadata."""

    id: str
    alias: str
    wav_path: Path
    sample_rate: int
    channel_count: int
    frame_count: int
    offset_ms: float
    consonant_ms: float
    cutoff_ms: float
    preutterance_ms: float
    overlap_ms: float
    sections: tuple[VoicebankSection, ...]

    @property
    def duration_seconds(self) -> float:
        return sum(section.duration_seconds for section in self.sections)


@dataclass(frozen=True, slots=True)
class VoicebankIssue:
    """A skipped voicebank entry that remains visible to callers."""

    oto_path: Path
    line_number: int
    message: str


@dataclass(frozen=True, slots=True)
class Voicebank:
    """A complete loaded voicebank."""

    root: Path
    units: tuple[VoicebankUnit, ...]
    fingerprint: str
    issues: tuple[VoicebankIssue, ...] = ()
    cache_hit: bool = False

    def units_for_alias(self, alias: str) -> tuple[VoicebankUnit, ...]:
        """Return every recording matching an alias, preserving bank order."""
        return tuple(unit for unit in self.units if unit.alias == alias)
