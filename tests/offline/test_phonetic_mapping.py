from pathlib import Path

import pytest

from synthstream.offline.phonetic import AikoEnglishAliasMap, RecognizedWord
from synthstream.voicebank import load_voicebank

PROJECT_ROOT = Path(__file__).parents[2]


def _aiko_map() -> AikoEnglishAliasMap:
    bank_path = PROJECT_ROOT / "voicebank" / "Kikyuune Aiko RockLoud CVVC EN"
    if not bank_path.is_dir():
        pytest.skip("local Aiko development voicebank is not installed")
    return AikoEnglishAliasMap(load_voicebank(bank_path))


def test_known_sentence_maps_to_existing_contextual_aiko_units() -> None:
    mapper = _aiko_map()
    text = "i had that curiosity beside me at this moment"
    words = tuple(
        RecognizedWord(word, index * 0.5, (index + 1) * 0.5, 0.9)
        for index, word in enumerate(text.split())
    )

    result = mapper.plan(words)
    aliases = [planned.alias for planned in result.aliases]

    assert result.unmapped_words == ()
    assert aliases[:5] == ["-I", "h@", "@d", "dh@", "@t"]
    assert {"kyo", "rE", "E9", "su", "ut", "tE"}.issubset(aliases)
    assert {"bi", "is", "sI", "Id", "mE", "-@", "dhi", "mO", "Om", "mu"}.issubset(
        aliases
    )
    assert all(planned.unit.alias == planned.alias for planned in result.aliases)
    assert all(planned.start_seconds < planned.end_seconds for planned in result.aliases)


def test_unknown_dictionary_word_is_reported_instead_of_fabricating_alias() -> None:
    mapper = _aiko_map()

    result = mapper.plan((RecognizedWord("synthstreamxyz", 0.0, 0.5, 0.8),))

    assert result.aliases == ()
    assert result.unmapped_words == ("synthstreamxyz",)
