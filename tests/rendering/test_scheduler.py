from pathlib import Path

import numpy as np
import soundfile as sf

from synthstream.rendering import RenderSegment, VoicebankRenderer, VoicebankRenderScheduler
from synthstream.voicebank import load_voicebank

SAMPLE_RATE = 16_000


def _make_bank(root: Path) -> None:
    time = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    for name, frequency in (("a.wav", 220.0), ("b.wav", 330.0)):
        sf.write(root / name, 0.3 * np.sin(2 * np.pi * frequency * time), SAMPLE_RATE)
    (root / "oto.ini").write_text(
        "a.wav=a,0,200,0,50,20\n"
        "b.wav=b,0,200,0,50,20\n",
        encoding="utf-8",
    )


def _segment(unit_id: str, alias: str, index: int, start: float, end: float) -> RenderSegment:
    section_kind = ("onset", "transition", "sustain")[index]
    return RenderSegment(unit_id, alias, index, section_kind, start, end)


def test_internal_sections_have_no_synthetic_crossfade_but_alias_onset_uses_oto_overlap(
    tmp_path: Path,
) -> None:
    _make_bank(tmp_path)
    bank = load_voicebank(tmp_path, use_cache=False)
    scheduler = VoicebankRenderScheduler(
        bank,
        VoicebankRenderer(),
        SAMPLE_RATE,
        staging_seconds=0.0,
    )
    first = bank.units[0]
    second = bank.units[1]

    assert scheduler.append(_segment(first.id, first.alias, 0, 0.0, 0.05)).overlap_samples == 0
    assert scheduler.append(_segment(first.id, first.alias, 1, 0.05, 0.20)).overlap_samples == 0
    assert scheduler.append(_segment(first.id, first.alias, 2, 0.20, 0.80)).overlap_samples == 0
    boundary = scheduler.append(_segment(second.id, second.alias, 0, 0.80, 0.85))

    assert boundary.overlap_samples == round(second.overlap_ms * SAMPLE_RATE / 1000.0)


def test_silence_resets_oto_overlap_context(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    bank = load_voicebank(tmp_path, use_cache=False)
    scheduler = VoicebankRenderScheduler(bank, VoicebankRenderer(), SAMPLE_RATE)
    first, second = bank.units

    scheduler.append(_segment(first.id, first.alias, 0, 0.0, 0.05))
    scheduler.append(
        RenderSegment(None, None, None, "silence", 0.05, 0.15)
    )
    result = scheduler.append(_segment(second.id, second.alias, 0, 0.15, 0.20))

    assert result.overlap_samples == 0


def test_live_append_does_not_backfill_elapsed_timeline_gap(tmp_path: Path) -> None:
    _make_bank(tmp_path)
    bank = load_voicebank(tmp_path, use_cache=False)
    scheduler = VoicebankRenderScheduler(
        bank,
        VoicebankRenderer(),
        SAMPLE_RATE,
        staging_seconds=0.0,
    )
    unit = bank.units[0]

    scheduler.append(_segment(unit.id, unit.alias, 0, 0.0, 0.05))
    result = scheduler.append(
        _segment(unit.id, unit.alias, 0, 1.0, 1.05),
        include_leading_gap=False,
    )

    assert len(result.released) == result.target_samples
    assert scheduler.scheduled_samples == round(1.05 * SAMPLE_RATE)


def test_live_gap_limit_keeps_bounded_queue_from_filling_with_old_silence(
    tmp_path: Path,
) -> None:
    _make_bank(tmp_path)
    bank = load_voicebank(tmp_path, use_cache=False)
    scheduler = VoicebankRenderScheduler(
        bank,
        VoicebankRenderer(),
        SAMPLE_RATE,
        staging_seconds=0.0,
    )
    unit = bank.units[0]

    scheduler.append(_segment(unit.id, unit.alias, 0, 0.0, 0.05))
    result = scheduler.append(
        _segment(unit.id, unit.alias, 0, 1.0, 1.05),
        include_leading_gap=True,
        leading_gap_limit_samples=100,
    )

    assert len(result.released) < round(0.2 * SAMPLE_RATE)
    assert scheduler.scheduled_samples == round(1.05 * SAMPLE_RATE)
