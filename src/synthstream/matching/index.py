"""Precompute and cache matcher features for every voicebank section."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from synthstream.analysis import AnalysisConfig, FeatureExtractor
from synthstream.voicebank import Voicebank

FloatArray = npt.NDArray[np.float32]
_CACHE_SCHEMA = 1


@dataclass(frozen=True, slots=True)
class SectionTemplate:
    """One cached voicebank-section feature trajectory."""

    unit_id: str
    alias: str
    section_index: int
    section_kind: str
    nominal_duration_seconds: float
    nominal_frame_count: int
    minimum_stretch: float
    maximum_stretch: float
    duration_penalty_strength: float
    features: FloatArray
    start_signature: FloatArray

    @property
    def id(self) -> str:
        return f"{self.unit_id}#{self.section_index}"


class SectionFeatureIndex:
    """All selectable voicebank sections and their precomputed features."""

    def __init__(
        self,
        templates: tuple[SectionTemplate, ...],
        config: AnalysisConfig,
        bank_fingerprint: str,
        *,
        cache_hit: bool = False,
    ) -> None:
        if not templates:
            raise ValueError("section feature index cannot be empty")
        self.templates = templates
        self.config = config
        self.bank_fingerprint = bank_fingerprint
        self.cache_hit = cache_hit
        self._signatures = np.stack([template.start_signature for template in templates])

    @classmethod
    def build(
        cls,
        bank: Voicebank,
        extractor: FeatureExtractor | None = None,
        *,
        use_cache: bool = True,
    ) -> SectionFeatureIndex:
        """Analyze every section, or restore unchanged trajectories from cache."""
        feature_extractor = extractor or FeatureExtractor()
        config = feature_extractor.config
        cache_path = _cache_path(bank.root, config)
        if use_cache:
            cached = _read_cache(cache_path, config, bank.fingerprint)
            if cached is not None:
                return cls(cached, config, bank.fingerprint, cache_hit=True)

        templates: list[SectionTemplate] = []
        for unit in bank.units:
            for section_index, section in enumerate(unit.sections):
                waveform, sample_rate = sf.read(
                    unit.wav_path,
                    start=section.start_sample,
                    stop=section.end_sample,
                    dtype="float32",
                    always_2d=True,
                )
                mono = np.asarray(np.mean(waveform, axis=1), dtype=np.float32)
                if sample_rate != config.sample_rate:
                    divisor = math.gcd(sample_rate, config.sample_rate)
                    mono = np.asarray(
                        resample_poly(
                            mono,
                            config.sample_rate // divisor,
                            sample_rate // divisor,
                        ),
                        dtype=np.float32,
                    )
                features = feature_extractor.analyze(mono).recognition_features
                if not len(features):
                    continue
                minimum, maximum, penalty = _duration_model(section.kind)
                nominal_frames = max(
                    1, round(section.duration_seconds * config.sample_rate / config.hop_samples)
                )
                templates.append(
                    SectionTemplate(
                        unit.id,
                        unit.alias,
                        section_index,
                        section.kind,
                        section.duration_seconds,
                        nominal_frames,
                        minimum,
                        maximum,
                        penalty,
                        features,
                        np.mean(features[: min(3, len(features))], axis=0).astype(np.float32),
                    )
                )
        result = tuple(templates)
        if not result:
            raise ValueError("voicebank contains no analyzable sections")
        if use_cache:
            _write_cache(cache_path, result, config, bank.fingerprint)
        return cls(result, config, bank.fingerprint)

    def rank_start_candidates(
        self, human_features: FloatArray, *, limit: int | None = None
    ) -> tuple[SectionTemplate, ...]:
        """Rank all sections by a cheap vectorized onset comparison."""
        query = np.asarray(human_features, dtype=np.float32)
        if query.shape != (self._signatures.shape[1],):
            raise ValueError("human feature vector has the wrong shape")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive or None")
        distances = np.mean(np.square(self._signatures - query), axis=1)
        order = np.argsort(distances, kind="stable")
        if limit is not None:
            order = order[:limit]
        return tuple(self.templates[int(index)] for index in order)


def _duration_model(section_kind: str) -> tuple[float, float, float]:
    if section_kind == "onset":
        return 0.55, 1.8, 1.5
    if section_kind == "transition":
        return 0.5, 2.0, 1.0
    return 0.35, 3.0, 0.35


def _cache_path(root: Path, config: AnalysisConfig) -> Path:
    config_json = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(config_json.encode()).hexdigest()[:12]
    return root / ".synthstream-cache" / f"section-features-v1-{digest}.npz"


def _write_cache(
    path: Path,
    templates: tuple[SectionTemplate, ...],
    config: AnalysisConfig,
    bank_fingerprint: str,
) -> None:
    metadata: dict[str, Any] = {
        "schema": _CACHE_SCHEMA,
        "config": asdict(config),
        "bank_fingerprint": bank_fingerprint,
        "templates": [],
    }
    arrays: dict[str, FloatArray] = {}
    for index, template in enumerate(templates):
        feature_key = f"features_{index}"
        signature_key = f"signature_{index}"
        arrays[feature_key] = template.features
        arrays[signature_key] = template.start_signature
        metadata["templates"].append(
            {
                "unit_id": template.unit_id,
                "alias": template.alias,
                "section_index": template.section_index,
                "section_kind": template.section_kind,
                "nominal_duration_seconds": template.nominal_duration_seconds,
                "nominal_frame_count": template.nominal_frame_count,
                "minimum_stretch": template.minimum_stretch,
                "maximum_stretch": template.maximum_stretch,
                "duration_penalty_strength": template.duration_penalty_strength,
                "feature_key": feature_key,
                "signature_key": signature_key,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        metadata=np.array(json.dumps(metadata)),
        **arrays,  # type: ignore[arg-type]
    )
    temporary.replace(path)


def _read_cache(
    path: Path, config: AnalysisConfig, bank_fingerprint: str
) -> tuple[SectionTemplate, ...] | None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if (
                metadata["schema"] != _CACHE_SCHEMA
                or metadata["config"] != asdict(config)
                or metadata["bank_fingerprint"] != bank_fingerprint
            ):
                return None
            templates = []
            for item in metadata["templates"]:
                feature_key = item.pop("feature_key")
                signature_key = item.pop("signature_key")
                templates.append(
                    SectionTemplate(
                        **item,
                        features=np.asarray(archive[feature_key], dtype=np.float32),
                        start_signature=np.asarray(archive[signature_key], dtype=np.float32),
                    )
                )
            return tuple(templates)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
