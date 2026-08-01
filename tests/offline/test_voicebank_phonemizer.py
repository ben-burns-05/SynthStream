from pathlib import Path

import pytest

from synthstream.offline.direct_phonemes import DetectedPhone, DirectAliasPlanner
from synthstream.offline.voicebank_phonemizer import (
    detect_voicebank_profile,
    read_presamp_metadata,
)
from synthstream.voicebank import load_voicebank

PROJECT_ROOT = Path(__file__).parents[2]
SENTENCE_PHONES = (
    "aɪ", "h", "æ", "d", "ð", "æ", "t", "k", "j", "uː", "ɹ", "ɪ", "ɑː", "s",
    "ɪ", "ɾ", "i", "b", "ɪ", "s", "aɪ", "d", "m", "i", "æ", "t", "ð", "ɪ", "s",
    "m", "oʊ", "m", "ə", "n", "t",
)


@pytest.mark.parametrize(
    ("directory", "profile_name"),
    (
        ("Kikyuune Aiko RockLoud CVVC EN", "aiko-cvvc"),
        ("CZloid VCCV 2015", "english-vccv"),
        ("TETO-English-150401", "english-presamp"),
    ),
)
def test_real_bank_profile_resolves_direct_phones_to_existing_aliases(
    directory: str, profile_name: str
) -> None:
    bank_path = PROJECT_ROOT / "voicebank" / directory
    if not bank_path.is_dir():
        pytest.skip(f"local development voicebank is not installed: {directory}")

    bank = load_voicebank(bank_path)
    capability = detect_voicebank_profile(bank)
    assert capability.profile is not None
    assert capability.profile.name == profile_name
    assert capability.supported
    assert capability.alias_coverage >= 0.9

    phones = tuple(
        DetectedPhone(phone, index * 0.1, (index + 1) * 0.1, 0.9)
        for index, phone in enumerate(SENTENCE_PHONES)
    )
    recognition = DirectAliasPlanner(bank, capability.profile).plan(phones)
    aliases = [planned.alias for planned in recognition.aliases]

    assert recognition.unmapped_phones == ()
    assert aliases
    assert all(alias in {unit.alias for unit in bank.units} for alias in aliases)
    if profile_name == "aiko-cvvc":
        assert aliases[:5] == ["-I", "h@", "@d", "dh@", "@t"]
    elif profile_name == "english-vccv":
        assert "@ d" in aliases
    else:
        assert aliases[:3] == ["- aI", "aI h{", "{ d"]


def test_teto_presamp_metadata_is_read_as_declarative_profile_data() -> None:
    path = next(
        (PROJECT_ROOT / "voicebank" / "TETO-English-150401").rglob("presamp.ini"),
        None,
    )
    if path is None:
        pytest.skip("TETO English presamp.ini is not installed")
    metadata = read_presamp_metadata(path)
    assert {"aI", "{", "@"}.issubset(metadata["VOWEL"])
    assert {"h", "d", "D", "tS"}.issubset(metadata["CONSONANT"])
    assert metadata["PRIORITY"][:4] == ("k", "g", "t", "d")


def test_japanese_bank_is_not_mislabelled_as_english() -> None:
    bank_path = PROJECT_ROOT / "voicebank" / "TETO-namerakaongen211125"
    if not bank_path.is_dir():
        pytest.skip("local Japanese development voicebank is not installed")
    capability = detect_voicebank_profile(load_voicebank(bank_path))
    assert not capability.supported
    assert capability.profile is None


@pytest.mark.parametrize(
    ("directory", "mode"),
    (
        ("CZloid VCCV 2015", "wav2vec2-ipa-ctc-english-vccv"),
        ("TETO-English-150401", "wav2vec2-ipa-ctc-english-presamp"),
    ),
)
def test_real_english_bank_runs_direct_timeline(
    directory: str, mode: str
) -> None:
    bank_path = PROJECT_ROOT / "voicebank" / directory
    fixture = PROJECT_ROOT / "tests" / "fixtures" / "human" / "voices_sentence.wav"
    if not bank_path.is_dir():
        pytest.skip(f"local development voicebank is not installed: {directory}")

    from synthstream.offline import OfflineRecognizer

    timeline = OfflineRecognizer.from_voicebank(bank_path).recognize(fixture)
    voiced = [segment for segment in timeline.segments if not segment.silence]
    assert timeline.recognition_mode == mode
    assert timeline.voicebank_profile is not None
    assert timeline.alias_coverage >= 0.9
    assert timeline.unmapped_phonemes == ()
    assert len({segment.alias for segment in voiced}) >= 20
    assert all(segment.alias is not None for segment in voiced)
