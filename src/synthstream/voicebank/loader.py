"""Load complete UTAU ``oto.ini`` voicebanks with metadata caching."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from synthstream.analysis import estimate_quantized_pitch_hz
from synthstream.voicebank.models import (
    Voicebank,
    VoicebankIssue,
    VoicebankSection,
    VoicebankUnit,
)

_CACHE_SCHEMA = 3
_CACHE_DIRECTORY = ".synthstream-cache"
_CACHE_FILENAME = "voicebank-v1.json"


class VoicebankLoadError(ValueError):
    """Raised when a voicebank cannot be represented safely."""


def load_voicebank(
    root: str | Path, *, use_cache: bool = True, strict: bool = False
) -> Voicebank:
    """Load every entry in every recursively discovered ``oto.ini`` file."""
    bank_root = Path(root).expanduser().resolve()
    if not bank_root.is_dir():
        raise VoicebankLoadError(f"Voicebank directory does not exist: {bank_root}")

    oto_paths = tuple(sorted(bank_root.rglob("oto.ini"), key=_relative_sort_key(bank_root)))
    if not oto_paths:
        raise VoicebankLoadError(f"No oto.ini files found below: {bank_root}")

    cache_path = bank_root / _CACHE_DIRECTORY / _CACHE_FILENAME
    if use_cache:
        cached = _load_cache_if_current(bank_root, oto_paths, cache_path, strict=strict)
        if cached is not None:
            return cached

    units: list[VoicebankUnit] = []
    issues: list[VoicebankIssue] = []
    source_paths: set[Path] = set(oto_paths)
    for oto_path in oto_paths:
        for line_number, line in _meaningful_lines(oto_path):
            referenced_path = _referenced_wav_path(bank_root, oto_path, line)
            if referenced_path is not None:
                source_paths.add(referenced_path)
            try:
                unit = _parse_unit(bank_root, oto_path, line_number, line)
            except VoicebankLoadError as error:
                if strict:
                    raise
                issues.append(VoicebankIssue(oto_path, line_number, str(error)))
                continue
            units.append(unit)
            source_paths.add(unit.wav_path)

    if not units:
        raise VoicebankLoadError(f"No usable oto.ini entries found below: {bank_root}")

    fingerprint, source_records = _fingerprint(bank_root, source_paths)
    bank = Voicebank(bank_root, tuple(units), fingerprint, tuple(issues))
    if use_cache:
        _write_cache(cache_path, bank, source_records)
    return bank


def _relative_sort_key(root: Path) -> Callable[[Path], str]:
    return lambda path: path.relative_to(root).as_posix().casefold()


def _meaningful_lines(path: Path) -> Iterable[tuple[int, str]]:
    raw = path.read_bytes()
    text: str | None = None
    for encoding in ("utf-8-sig", "cp932", "shift_jis"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise VoicebankLoadError(f"Unsupported oto.ini encoding: {path}")

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if line and not line.startswith(("#", ";")):
            yield line_number, line


def _parse_unit(root: Path, oto_path: Path, line_number: int, line: str) -> VoicebankUnit:
    location = f"{oto_path}:{line_number}"
    if "=" not in line:
        raise VoicebankLoadError(f"Missing '=' in oto.ini entry at {location}")
    wav_name, values = line.split("=", 1)
    fields = [field.strip() for field in values.split(",")]
    if len(fields) != 6:
        raise VoicebankLoadError(f"Expected 6 comma-separated fields at {location}")

    alias = fields[0] or Path(wav_name).stem
    timings = tuple(_parse_finite_float(value, location) for value in fields[1:])
    offset_ms, consonant_ms, cutoff_ms, preutterance_ms, overlap_ms = timings
    if offset_ms < 0 or consonant_ms < 0 or preutterance_ms < 0:
        raise VoicebankLoadError(
            f"Offset, consonant and preutterance must be non-negative at {location}"
        )

    normalized_name = wav_name.strip().replace("\\", os.sep).replace("/", os.sep)
    wav_path = (oto_path.parent / normalized_name).resolve()
    try:
        wav_path.relative_to(root)
    except ValueError as error:
        raise VoicebankLoadError(f"WAV path escapes voicebank root at {location}") from error
    if not wav_path.is_file():
        raise VoicebankLoadError(f"Referenced WAV does not exist at {location}: {wav_path}")

    try:
        info = sf.info(wav_path)
    except (RuntimeError, sf.LibsndfileError) as error:
        raise VoicebankLoadError(f"Cannot read WAV at {location}: {wav_path}") from error
    if info.samplerate <= 0 or info.frames <= 0:
        raise VoicebankLoadError(f"Referenced WAV is empty or invalid at {location}: {wav_path}")

    sections = _make_sections(
        info.frames,
        info.samplerate,
        offset_ms,
        consonant_ms,
        cutoff_ms,
        preutterance_ms,
        location,
    )
    source_pitch_hz = _estimate_source_pitch(wav_path, sections, info.samplerate)
    unit_id = f"{oto_path.relative_to(root).as_posix()}:{line_number}"
    return VoicebankUnit(
        id=unit_id,
        alias=alias,
        wav_path=wav_path,
        sample_rate=info.samplerate,
        channel_count=info.channels,
        frame_count=info.frames,
        offset_ms=offset_ms,
        consonant_ms=consonant_ms,
        cutoff_ms=cutoff_ms,
        preutterance_ms=preutterance_ms,
        overlap_ms=overlap_ms,
        sections=sections,
        source_pitch_hz=source_pitch_hz,
    )


def _referenced_wav_path(root: Path, oto_path: Path, line: str) -> Path | None:
    if "=" not in line:
        return None
    wav_name = line.split("=", 1)[0].strip().replace("\\", os.sep).replace("/", os.sep)
    path = (oto_path.parent / wav_name).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None
    return path


def _parse_finite_float(value: str, location: str) -> float:
    if not value:
        return 0.0
    try:
        number = float(value)
    except ValueError as error:
        raise VoicebankLoadError(f"Invalid timing value {value!r} at {location}") from error
    if not math.isfinite(number):
        raise VoicebankLoadError(f"Non-finite timing value {value!r} at {location}")
    return number


def _make_sections(
    frame_count: int,
    sample_rate: int,
    offset_ms: float,
    consonant_ms: float,
    cutoff_ms: float,
    preutterance_ms: float,
    location: str,
) -> tuple[VoicebankSection, ...]:
    def to_sample(milliseconds: float) -> int:
        return round(milliseconds * sample_rate / 1000)

    start = to_sample(offset_ms)
    end = frame_count - to_sample(cutoff_ms) if cutoff_ms >= 0 else start - to_sample(cutoff_ms)
    end = min(end, frame_count)
    if not 0 <= start < end <= frame_count:
        raise VoicebankLoadError(f"oto.ini timing selects no usable WAV audio at {location}")

    preutterance = min(max(start + to_sample(preutterance_ms), start), end)
    consonant = min(max(start + to_sample(consonant_ms), preutterance), end)
    boundaries = (
        ("onset", start, preutterance),
        ("transition", preutterance, consonant),
        ("sustain", consonant, end),
    )
    sections = tuple(
        VoicebankSection(kind, section_start, section_end, sample_rate)
        for kind, section_start, section_end in boundaries
        if section_end > section_start
    )
    if not sections:
        raise VoicebankLoadError(f"oto.ini timing creates no sections at {location}")
    return sections


def _estimate_source_pitch(
    wav_path: Path,
    sections: tuple[VoicebankSection, ...],
    sample_rate: int,
) -> float | None:
    """Precompute the canonical recorded pitch used for alias transfer."""
    reference = next(
        (section for section in sections if section.kind == "sustain"),
        sections[-1],
    )
    try:
        waveform, file_sample_rate = sf.read(
            wav_path,
            start=reference.start_sample,
            stop=reference.end_sample,
            dtype="float32",
            always_2d=True,
        )
    except (RuntimeError, sf.LibsndfileError):
        return None
    if file_sample_rate != sample_rate or not len(waveform):
        return None
    mono = np.asarray(np.mean(waveform, axis=1), dtype=np.float32)
    return estimate_quantized_pitch_hz(mono, sample_rate)


def _fingerprint(
    root: Path, paths: Iterable[Path]
) -> tuple[str, list[dict[str, bool | int | str]]]:
    records: list[dict[str, bool | int | str]] = []
    for path in sorted(paths, key=_relative_sort_key(root)):
        record: dict[str, bool | int | str] = {
            "path": path.relative_to(root).as_posix(),
            "exists": path.is_file(),
        }
        if path.is_file():
            stat = path.stat()
            record.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        records.append(record)
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), records


def _load_cache_if_current(
    root: Path, oto_paths: tuple[Path, ...], cache_path: Path, *, strict: bool
) -> Voicebank | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("schema") != _CACHE_SCHEMA:
            return None
        records = payload["sources"]
        cached_oto = {
            record["path"] for record in records if Path(record["path"]).name == "oto.ini"
        }
        current_oto = {path.relative_to(root).as_posix() for path in oto_paths}
        if cached_oto != current_oto:
            return None
        source_paths = tuple(root / record["path"] for record in records)
        fingerprint, current_records = _fingerprint(root, source_paths)
        if current_records != records or fingerprint != payload["fingerprint"]:
            return None
        issues = tuple(
            VoicebankIssue(
                root / item["oto_path"], item["line_number"], item["message"]
            )
            for item in payload.get("issues", [])
        )
        if strict and issues:
            return None
        units = tuple(_unit_from_dict(root, item) for item in payload["units"])
        return Voicebank(root, units, fingerprint, issues, cache_hit=True)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_cache(
    cache_path: Path,
    bank: Voicebank,
    records: list[dict[str, bool | int | str]],
) -> None:
    payload = {
        "schema": _CACHE_SCHEMA,
        "fingerprint": bank.fingerprint,
        "sources": records,
        "units": [_unit_to_dict(bank.root, unit) for unit in bank.units],
        "issues": [
            {
                "oto_path": issue.oto_path.relative_to(bank.root).as_posix(),
                "line_number": issue.line_number,
                "message": issue.message,
            }
            for issue in bank.issues
        ],
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(cache_path)


def _unit_to_dict(root: Path, unit: VoicebankUnit) -> dict[str, Any]:
    return {
        "id": unit.id,
        "alias": unit.alias,
        "wav_path": unit.wav_path.relative_to(root).as_posix(),
        "sample_rate": unit.sample_rate,
        "channel_count": unit.channel_count,
        "frame_count": unit.frame_count,
        "offset_ms": unit.offset_ms,
        "consonant_ms": unit.consonant_ms,
        "cutoff_ms": unit.cutoff_ms,
        "preutterance_ms": unit.preutterance_ms,
        "overlap_ms": unit.overlap_ms,
        "source_pitch_hz": unit.source_pitch_hz,
        "sections": [
            {
                "kind": section.kind,
                "start_sample": section.start_sample,
                "end_sample": section.end_sample,
                "sample_rate": section.sample_rate,
            }
            for section in unit.sections
        ],
    }


def _unit_from_dict(root: Path, item: dict[str, Any]) -> VoicebankUnit:
    sections = tuple(VoicebankSection(**section) for section in item.pop("sections"))
    wav_path = (root / item.pop("wav_path")).resolve()
    return VoicebankUnit(wav_path=wav_path, sections=sections, **item)
