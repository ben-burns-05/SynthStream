import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from synthstream.offline import direct_phonemes


def test_prepare_direct_window_pads_but_preserves_valid_audio_range() -> None:
    samples = np.ones(4_000, dtype=np.float32)

    window = direct_phonemes.prepare_direct_window(samples, 16_000)

    assert len(window.samples) == round(0.8 * 16_000)
    assert window.valid_start_seconds == 0.0
    assert window.valid_end_seconds == pytest.approx(0.25)
    assert np.array_equal(window.samples[: len(samples)], samples)
    assert np.all(window.samples[len(samples) :] == 0)


def test_canonicalize_direct_audio_mixes_stereo_and_resamples() -> None:
    source = np.column_stack(
        (
            np.ones(2_205, dtype=np.float32),
            np.zeros(2_205, dtype=np.float32),
        )
    )

    canonical = direct_phonemes.canonicalize_direct_audio(source, 22_050)

    assert canonical.dtype == np.float32
    assert len(canonical) == pytest.approx(1_600, abs=2)
    assert float(np.mean(canonical)) == pytest.approx(0.5, abs=0.01)


def test_stable_phone_indices_require_an_ordered_stable_prefix() -> None:
    previous = (
        direct_phonemes.DetectedPhone("a", 0.00, 0.10, 0.9),
        direct_phonemes.DetectedPhone("b", 0.10, 0.20, 0.9),
        direct_phonemes.DetectedPhone("a", 0.20, 0.30, 0.9),
    )
    current = (
        direct_phonemes.DetectedPhone("a", 0.01, 0.11, 0.9),
        direct_phonemes.DetectedPhone("b", 0.11, 0.21, 0.9),
        direct_phonemes.DetectedPhone("a", 0.21, 0.31, 0.9),
    )

    assert direct_phonemes.stable_phone_indices(previous, current) == (0, 1, 2)


def test_stable_phone_indices_stop_at_the_first_changed_phone() -> None:
    previous = (
        direct_phonemes.DetectedPhone("a", 0.00, 0.10, 0.9),
        direct_phonemes.DetectedPhone("b", 0.10, 0.20, 0.9),
    )
    current = (
        direct_phonemes.DetectedPhone("e", 0.00, 0.10, 0.9),
        direct_phonemes.DetectedPhone("b", 0.11, 0.21, 0.9),
    )

    assert direct_phonemes.stable_phone_indices(previous, current) == ()


def test_direct_ipa_uses_canonical_feature_extraction_and_clips_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vocabulary_path = tmp_path / "vocab.json"
    vocabulary_path.write_text(json.dumps({"<pad>": 0, "a": 1}), encoding="utf-8")
    extracted: list[np.ndarray] = []
    received_masks: list[torch.Tensor | None] = []

    class DummyModel:
        def __call__(self, input_values: torch.Tensor, **kwargs: object) -> object:
            received_masks.append(kwargs.get("attention_mask"))  # type: ignore[arg-type]
            assert input_values.shape == (1, round(0.8 * 16_000))
            logits = torch.tensor([[[0.0, 8.0], [0.0, 8.0], [8.0, 0.0]]])
            return SimpleNamespace(logits=logits)

    class DummyExtractor:
        def __call__(
            self,
            waveform: np.ndarray,
            *,
            sampling_rate: int,
            return_tensors: str,
            return_attention_mask: bool,
        ) -> object:
            assert sampling_rate == 16_000
            assert return_tensors == "pt"
            assert return_attention_mask
            extracted.append(np.asarray(waveform).copy())
            input_values = torch.from_numpy(np.asarray(waveform)).unsqueeze(0)
            attention_mask = torch.ones_like(input_values, dtype=torch.long)
            return SimpleNamespace(
                input_values=input_values,
                attention_mask=attention_mask,
            )

    monkeypatch.setattr(
        direct_phonemes,
        "_load_model_assets",
        lambda: (DummyModel(), str(vocabulary_path)),
    )
    monkeypatch.setattr(
        direct_phonemes,
        "_load_feature_extractor",
        lambda: DummyExtractor(),
    )

    source = np.ones(4_000, dtype=np.float32)
    phones = direct_phonemes.DirectIPARecognizer().recognize(source, 16_000)

    assert len(extracted) == 1
    assert len(extracted[0]) == round(0.8 * 16_000)
    assert np.array_equal(extracted[0][: len(source)], source)
    assert len(received_masks) == 1
    assert received_masks[0] is not None
    assert len(phones) == 1
    assert phones[0].ipa == "a"
    assert phones[0].start_seconds == 0.0
    assert phones[0].end_seconds == pytest.approx(0.25)


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

    extractor_calls = 0

    def fake_extractor() -> object:
        nonlocal extractor_calls
        extractor_calls += 1
        return object()

    monkeypatch.setattr(direct_phonemes, "_load_model_assets", fake_loader)
    monkeypatch.setattr(direct_phonemes, "_load_feature_extractor", fake_extractor)
    recognizer = direct_phonemes.DirectIPARecognizer()

    recognizer.warmup()
    recognizer.warmup()

    assert calls == 1
    assert extractor_calls == 1
