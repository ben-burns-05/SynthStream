from pathlib import Path

import pytest

from synthstream.offline import OfflineRecognizer

PROJECT_ROOT = Path(__file__).parents[2]


def test_recorded_english_produces_semantic_real_aiko_alias_timeline() -> None:
    bank_path = PROJECT_ROOT / "voicebank" / "Kikyuune Aiko RockLoud CVVC EN"
    if not bank_path.is_dir():
        pytest.skip("local Aiko development voicebank is not installed")

    recognizer = OfflineRecognizer.from_voicebank(bank_path)
    timeline = recognizer.recognize(
        PROJECT_ROOT / "tests" / "fixtures" / "human" / "voices_excerpt.wav"
    )
    voiced = [segment for segment in timeline.segments if not segment.silence]
    aliases = list(dict.fromkeys(segment.alias for segment in voiced))

    assert timeline.recognition_mode == "wav2vec2-ctc-cmudict-aiko-cvvc"
    assert timeline.transcript == "i had that"
    assert timeline.unmapped_words == ()
    assert aliases == ["-I", "h@", "@d", "dh@", "@t"]
    assert all(segment.unit_id is not None for segment in voiced)
    bank_aliases = {unit.alias for unit in recognizer.bank.units}
    assert all(segment.alias in bank_aliases for segment in voiced)
    assert timeline.segments[0].start_seconds == 0
    assert timeline.segments[-1].end_seconds == pytest.approx(0.5, abs=0.01)
    assert timeline.segments[-1].silence
    assert timeline.segments[-1].start_seconds == pytest.approx(0.4, abs=0.04)
    assert all(
        left.end_seconds == pytest.approx(right.start_seconds, abs=1e-6)
        for left, right in zip(timeline.segments, timeline.segments[1:], strict=False)
    )
    assert all(segment.duration_seconds > 0 for segment in timeline.segments)

    payload = timeline.to_dict()
    assert payload["transcript"] == "i had that"
    assert payload["recognition_mode"] == "wav2vec2-ctc-cmudict-aiko-cvvc"
