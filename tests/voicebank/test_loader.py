from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synthstream.voicebank import VoicebankLoadError, load_voicebank
from synthstream.voicebank import loader as loader_module

SAMPLE_RATE = 16_000


def _write_wav(path: Path, duration_ms: int = 500) -> None:
    frame_count = round(duration_ms * SAMPLE_RATE / 1000)
    waveform = np.linspace(-0.1, 0.1, frame_count, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, SAMPLE_RATE, subtype="PCM_16")


def _write_oto(path: Path, contents: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents.encode(encoding))


def test_loads_complete_recursive_voicebank_and_builds_sections(tmp_path: Path) -> None:
    _write_wav(tmp_path / "a.wav")
    _write_wav(tmp_path / "sub" / "ka.wav", duration_ms=400)
    _write_oto(tmp_path / "oto.ini", "a.wav=a,10,120,20,40,15\n")
    _write_oto(
        tmp_path / "sub" / "oto.ini",
        "# Japanese alias in a CP932 metadata file\nka.wav=か,20,100,-300,30,-5\n",
        "cp932",
    )

    bank = load_voicebank(tmp_path)

    assert len(bank.units) == 2
    assert tuple(unit.alias for unit in bank.units) == ("a", "か")
    assert bank.units_for_alias("か") == (bank.units[1],)
    assert tuple(section.kind for section in bank.units[0].sections) == (
        "onset",
        "transition",
        "sustain",
    )
    assert bank.units[0].sections[0].start_sample == 160
    assert bank.units[0].sections[-1].end_sample == 7_680
    assert bank.units[1].sections[-1].end_sample == 5_120
    assert bank.units[0].sample_rate == SAMPLE_RATE
    assert bank.units[0].channel_count == 1
    assert bank.units[0].duration_seconds == pytest.approx(0.47)
    assert not bank.cache_hit


def test_reuses_cache_without_reopening_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_wav(tmp_path / "a.wav")
    _write_oto(tmp_path / "oto.ini", "a.wav=a,0,100,0,30,10\n")
    first = load_voicebank(tmp_path)

    def fail_if_called(path: Path) -> None:
        pytest.fail(f"cache hit reopened WAV header: {path}")

    monkeypatch.setattr(loader_module.sf, "info", fail_if_called)
    second = load_voicebank(tmp_path)

    assert second.cache_hit
    assert second.fingerprint == first.fingerprint
    assert second.units == first.units


def test_cache_invalidates_when_metadata_changes(tmp_path: Path) -> None:
    _write_wav(tmp_path / "a.wav")
    oto_path = tmp_path / "oto.ini"
    _write_oto(oto_path, "a.wav=old,0,100,0,30,10\n")
    load_voicebank(tmp_path)
    _write_oto(oto_path, "a.wav=new alias,0,100,0,30,10\n")

    reloaded = load_voicebank(tmp_path)

    assert not reloaded.cache_hit
    assert reloaded.units[0].alias == "new alias"


@pytest.mark.parametrize(
    ("oto_entry", "message"),
    [
        ("broken entry", "Missing '='"),
        ("missing.wav=a,0,100,0,30,10", "does not exist"),
        ("a.wav=a,nan,100,0,30,10", "Non-finite"),
        ("a.wav=a,600,100,0,30,10", "selects no usable"),
    ],
)
def test_reports_invalid_entries(tmp_path: Path, oto_entry: str, message: str) -> None:
    _write_wav(tmp_path / "a.wav")
    _write_oto(tmp_path / "oto.ini", f"{oto_entry}\n")

    with pytest.raises(VoicebankLoadError, match=message):
        load_voicebank(tmp_path, use_cache=False)


def test_rejects_directory_without_metadata(tmp_path: Path) -> None:
    with pytest.raises(VoicebankLoadError, match="No oto.ini"):
        load_voicebank(tmp_path)
