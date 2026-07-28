from pathlib import Path

import pytest

from synthstream.voicebank import load_voicebank

PROJECT_ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("directory_name", "minimum_units"),
    [
        ("CZloid VCCV 2015", 3_000),
        ("Kikyuune Aiko RockLoud CVVC EN", 1_500),
        ("TETO-English-150401", 2_600),
        ("TETO-namerakaongen211125", 300),
    ],
)
def test_installed_real_voicebank_loads_completely_and_reuses_cache(
    directory_name: str, minimum_units: int
) -> None:
    bank_path = PROJECT_ROOT / "voicebank" / directory_name
    if not bank_path.is_dir():
        pytest.skip(f"local development voicebank is not installed: {directory_name}")

    first = load_voicebank(bank_path)
    cached = load_voicebank(bank_path)

    assert len(first.units) >= minimum_units
    assert cached.cache_hit
    assert cached.units == first.units
    assert cached.issues == first.issues
    assert all(unit.wav_path.is_file() for unit in cached.units)
    assert all(unit.alias for unit in cached.units)
    assert all(unit.sections for unit in cached.units)
    assert all(
        0 <= unit.sections[0].start_sample < unit.sections[-1].end_sample <= unit.frame_count
        for unit in cached.units
    )
