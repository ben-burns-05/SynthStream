import json
from pathlib import Path

import pytest

from synthstream.offline import direct_phonemes


def test_direct_ipa_warmup_loads_model_assets_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocabulary_path = tmp_path / "vocab.json"
    vocabulary_path.write_text(json.dumps({"<pad>": 0, "a": 1}), encoding="utf-8")
    calls = 0

    class DummyModel:
        def eval(self) -> "DummyModel":
            return self

    def fake_loader() -> tuple[DummyModel, str]:
        nonlocal calls
        calls += 1
        return DummyModel(), str(vocabulary_path)

    monkeypatch.setattr(direct_phonemes, "_load_model_assets", fake_loader)
    recognizer = direct_phonemes.DirectIPARecognizer()

    recognizer.warmup()
    recognizer.warmup()

    assert calls == 1
