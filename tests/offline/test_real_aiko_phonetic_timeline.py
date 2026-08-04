from pathlib import Path

import pytest

from synthstream.offline import OfflineRecognizer, synthesize_timeline

PROJECT_ROOT = Path(__file__).parents[2]


def test_recorded_english_produces_semantic_real_aiko_alias_timeline(tmp_path: Path) -> None:
    bank_path = PROJECT_ROOT / "voicebank" / "Kikyuune Aiko RockLoud CVVC EN"
    if not bank_path.is_dir():
        pytest.skip("local Aiko development voicebank is not installed")

    recognizer = OfflineRecognizer.from_voicebank(bank_path)
    timeline = recognizer.recognize(
        PROJECT_ROOT / "tests" / "fixtures" / "human" / "voices_sentence.wav"
    )
    synthesized = synthesize_timeline(timeline, recognizer.bank)
    output_path = synthesized.write_wav(tmp_path / "aiko-output.wav")
    voiced = [segment for segment in timeline.segments if not segment.silence]
    aliases = list(dict.fromkeys(segment.alias for segment in voiced))

    assert timeline.recognition_mode == "wav2vec2-ipa-ctc-aiko-cvvc"
    assert timeline.transcript is None
    assert timeline.detected_phonemes == (
        "aɪ", "h", "æ", "d", "ð", "æ", "t", "k", "j", "uː", "ɹ", "ɪ",
        "ɑː", "s", "ɪ", "ɾ", "i", "b", "ɪ", "s", "aɪ", "d", "m", "i",
        "æ", "t", "ð", "ɪ", "s", "m", "oʊ", "m", "ə", "n", "t",
    )
    assert timeline.unmapped_words == ()
    assert timeline.unmapped_phonemes == ()
    assert aliases[:5] == ["-I", "h@", "@d", "dh@", "@t"]
    assert len(aliases) >= 20
    assert "kyo" in aliases
    assert "yo" not in aliases
    assert all(segment.unit_id is not None for segment in voiced)
    bank_aliases = {unit.alias for unit in recognizer.bank.units}
    assert all(segment.alias in bank_aliases for segment in voiced)
    assert timeline.segments[0].start_seconds == 0
    assert timeline.segments[-1].end_seconds == pytest.approx(3.4, abs=0.01)
    assert timeline.segments[-1].silence
    assert timeline.segments[-1].start_seconds == pytest.approx(3.1, abs=0.04)
    assert all(
        left.end_seconds == pytest.approx(right.start_seconds, abs=1e-6)
        for left, right in zip(timeline.segments, timeline.segments[1:], strict=False)
    )
    assert all(segment.duration_seconds > 0 for segment in timeline.segments)
    first_unit_stretches = [segment.stretch_ratio for segment in voiced[:3]]
    assert len(set(round(value, 2) for value in first_unit_stretches)) > 1

    payload = timeline.to_dict()
    assert payload["transcript"] is None
    assert payload["detected_phonemes"] == list(timeline.detected_phonemes)
    assert payload["recognition_mode"] == "wav2vec2-ipa-ctc-aiko-cvvc"
    assert payload["voicebank_profile"] == "aiko-cvvc"
    assert payload["voicebank_profile_confidence"] >= 0.9
    assert payload["alias_coverage"] >= 0.9
    assert output_path.is_file()
    assert synthesized.sample_rate == 44_100
    assert synthesized.duration_seconds == pytest.approx(3.4, abs=0.01)
    assert synthesized.voiced_segments > 50
    assert synthesized.silence_segments >= 1
    assert synthesized.overlap_segments > 10
    assert synthesized.samples.dtype == "float32"
    assert synthesized.samples.max() > 0.05
