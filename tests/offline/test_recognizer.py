import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from synthstream.offline import OfflineRecognizer, recognize_wav
from synthstream.offline.__main__ import main


def _tone(frequency: float, duration: float, sample_rate: int) -> np.ndarray:
    time = np.arange(round(sample_rate * duration), dtype=np.float32) / sample_rate
    fundamental = np.sin(2 * np.pi * frequency * time)
    harmonics = 0.35 * np.sin(4 * np.pi * frequency * time)
    return (0.3 * (fundamental + harmonics)).astype(np.float32)


def _speech_like_unit(sample_rate: int) -> np.ndarray:
    return np.concatenate(
        (
            _tone(220, 0.05, sample_rate),
            _tone(330, 0.15, sample_rate),
            _tone(440, 0.30, sample_rate),
        )
    )


def _make_voicebank(root: Path) -> None:
    sf.write(root / "a.wav", _speech_like_unit(16_000), 16_000)
    sf.write(
        root / "i.wav",
        np.concatenate(
            (
                _tone(560, 0.05, 16_000),
                _tone(670, 0.15, 16_000),
                _tone(780, 0.30, 16_000),
            )
        ),
        16_000,
    )
    (root / "oto.ini").write_text(
        "a.wav=a,0,200,0,50,20\n"
        "i.wav=i,0,200,0,50,20\n",
        encoding="utf-8",
    )


def test_human_wav_runs_through_production_pipeline_and_exports_json(
    tmp_path: Path,
) -> None:
    bank_path = tmp_path / "bank"
    bank_path.mkdir()
    _make_voicebank(bank_path)
    human_path = tmp_path / "human.wav"
    source = _speech_like_unit(22_050)
    stereo = np.column_stack((source, source * 0.8))
    sf.write(human_path, stereo, 22_050)
    output_path = tmp_path / "results" / "timeline.json"

    timeline = recognize_wav(
        human_path,
        bank_path,
        output_json=output_path,
        use_cache=False,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    voiced = [segment for segment in timeline.segments if not segment.silence]
    assert [segment.alias for segment in voiced] == ["a", "a", "a"]
    assert [segment.section_kind for segment in voiced] == [
        "onset",
        "transition",
        "sustain",
    ]
    assert timeline.sample_rate == 16_000
    assert timeline.input_duration_seconds == pytest.approx(0.5, abs=1e-3)
    assert timeline.segments[0].start_seconds == 0
    assert timeline.segments[-1].end_seconds == pytest.approx(0.5, abs=1e-3)
    assert all(segment.duration_seconds > 0 for segment in timeline.segments)
    assert all(segment.stretch_ratio > 0 for segment in voiced)
    assert payload["format"] == "synthstream-timeline"
    assert payload["version"] == 1
    assert payload["segments"][0]["alias"] == "a"
    assert payload["diagnostics"]["frames_processed"] > 0


def test_recorded_human_speech_wav_produces_section_timeline(tmp_path: Path) -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "human" / "voices_excerpt.wav"
    recorded, sample_rate = sf.read(fixture, dtype="float32")
    assert sample_rate == 16_000
    bank_path = tmp_path / "recorded-bank"
    bank_path.mkdir()
    sf.write(bank_path / "recorded.wav", recorded, sample_rate)
    (bank_path / "oto.ini").write_text(
        "recorded.wav=recorded-human,0,200,0,50,20\n", encoding="utf-8"
    )

    timeline = recognize_wav(fixture, bank_path, use_cache=False)
    voiced = [segment for segment in timeline.segments if not segment.silence]

    assert [segment.alias for segment in voiced] == [
        "recorded-human",
        "recorded-human",
        "recorded-human",
    ]
    assert [segment.section_kind for segment in voiced] == [
        "onset",
        "transition",
        "sustain",
    ]
    assert timeline.segments[-1].end_seconds == pytest.approx(0.5, abs=1e-3)


def test_offline_timeline_places_leading_silence(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank"
    bank_path.mkdir()
    _make_voicebank(bank_path)
    human_path = tmp_path / "human-with-silence.wav"
    audio = np.concatenate(
        (np.zeros(round(0.15 * 16_000), dtype=np.float32), _speech_like_unit(16_000))
    )
    sf.write(human_path, audio, 16_000)
    recognizer = OfflineRecognizer.from_voicebank(bank_path, use_cache=False)

    timeline = recognizer.recognize(human_path)

    assert timeline.segments[0].silence
    assert timeline.segments[0].duration_seconds == pytest.approx(0.15, abs=0.04)
    assert any(segment.alias == "a" for segment in timeline.segments)
    assert timeline.segments[-1].end_seconds == pytest.approx(0.65, abs=1e-3)


def test_offline_command_line_entry_point_writes_timeline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bank_path = tmp_path / "bank"
    bank_path.mkdir()
    _make_voicebank(bank_path)
    human_path = tmp_path / "human.wav"
    sf.write(human_path, _speech_like_unit(16_000), 16_000)
    output_path = tmp_path / "timeline.json"

    exit_code = main([str(human_path), str(bank_path), "--output", str(output_path)])

    assert exit_code == 0
    assert output_path.is_file()
    assert "Recognized" in capsys.readouterr().out


def test_offline_recognizer_rejects_missing_wav(tmp_path: Path) -> None:
    bank_path = tmp_path / "bank"
    bank_path.mkdir()
    _make_voicebank(bank_path)
    recognizer = OfflineRecognizer.from_voicebank(bank_path, use_cache=False)

    with pytest.raises(ValueError, match="does not exist"):
        recognizer.recognize(tmp_path / "missing.wav")
