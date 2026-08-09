"""Inventory-aware phoneme-to-OTO alias resolution.

The direct acoustic frontend emits IPA phones.  This module is the deliberately
separate, voicebank-format layer that turns those phones into aliases that are
actually present in a singer's ``oto.ini`` inventory.  It follows the same
separation used by OpenUtau: one common interface, with small profiles for the
different CVVC/VCCV/Presamp conventions.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from synthstream.voicebank import Voicebank


@dataclass(frozen=True, slots=True)
class AliasCandidate:
    alias: str
    phone_indices: tuple[int, ...]
    rank: int


@dataclass(frozen=True, slots=True)
class VoicebankProfile:
    """Rules for one compatible alias convention."""

    name: str
    family: str
    ipa_symbols: dict[str, str]
    vowels: frozenset[str]

    def symbol_for(self, ipa: str) -> str | None:
        return self.ipa_symbols.get(ipa)

    def candidates(
        self,
        symbols: tuple[str | None, ...],
        vowel_indices: tuple[int, ...],
        aliases: frozenset[str],
    ) -> tuple[AliasCandidate, ...]:
        candidates: list[AliasCandidate] = []
        for vowel_number, vowel_index in enumerate(vowel_indices):
            previous_vowel = vowel_indices[vowel_number - 1] if vowel_number else -1
            onset_start = previous_vowel + 1
            onset_indices = tuple(
                index
                for index in range(onset_start, vowel_index + 1)
                if symbols[index] is not None
            )
            if onset_indices:
                candidates.extend(
                    _available(
                        self._onset_forms(
                            symbols,
                            vowel_index,
                            previous_vowel,
                            onset_indices,
                        ),
                        onset_indices,
                        aliases,
                    )
                )

            next_vowel = (
                vowel_indices[vowel_number + 1]
                if vowel_number + 1 < len(vowel_indices)
                else len(symbols)
            )
            consonant_index = next(
                (
                    index
                    for index in range(vowel_index + 1, next_vowel)
                    if symbols[index] is not None
                ),
                None,
            )
            if consonant_index is not None:
                indices = (vowel_index, consonant_index)
                candidates.extend(
                    _available(
                        self._vc_forms(symbols, vowel_index, consonant_index),
                        indices,
                        aliases,
                    )
                )
                cluster_indices = tuple(
                    index
                    for index in range(consonant_index, next_vowel)
                    if symbols[index] is not None
                )
                if len(cluster_indices) > 1:
                    cluster = tuple(symbols[index] or "" for index in cluster_indices)
                    candidates.extend(
                        _available(
                            (
                                "".join(cluster),
                                " ".join(cluster),
                                "".join(cluster) + "-",
                                " ".join(cluster) + "-",
                            ),
                            cluster_indices,
                            aliases,
                        )
                    )
        return _deduplicate(candidates)

    def _onset_forms(
        self,
        symbols: tuple[str | None, ...],
        vowel_index: int,
        previous_vowel: int,
        onset_indices: tuple[int, ...],
    ) -> tuple[str, ...]:
        tokens = tuple(symbols[index] or "" for index in onset_indices)
        groups = (
            (tokens,)
            if previous_vowel < 0
            else tuple(tokens[start:] for start in range(len(tokens)))
        )
        if self.family == "english-presamp":
            if previous_vowel < 0:
                joined = "".join(tokens)
                spaced = " ".join(tokens)
                return (f"- {spaced}", f"- {joined}", f"-{joined}", spaced, joined)
            previous = symbols[previous_vowel] or ""
            return tuple(
                alias
                for group in groups
                for alias in (
                    f"{previous} {' '.join(group)}",
                    f"{previous} {''.join(group)}",
                    " ".join(group),
                    "".join(group),
                )
            )
        if previous_vowel < 0:
            joined = "".join(tokens)
            spaced = " ".join(tokens)
            if self.family == "english-vccv":
                return (f"-{joined}", f"_{joined}", joined, f"- {spaced}", spaced)
            return (f"-{joined}", joined, f"- {spaced}", spaced)
        return tuple(
            alias
            for group in groups
            for alias in (
                "".join(group),
                " ".join(group),
                f"_{''.join(group)}",
                f"_{' '.join(group)}",
            )
        )

    def _vc_forms(
        self,
        symbols: tuple[str | None, ...],
        vowel_index: int,
        consonant_index: int,
    ) -> tuple[str, ...]:
        vowel = symbols[vowel_index] or ""
        consonant = symbols[consonant_index] or ""
        if self.family == "english-presamp":
            return (f"{vowel} {consonant}", f"{vowel}{consonant}")
        if self.family == "english-vccv":
            return (f"{vowel} {consonant}", f"{vowel}{consonant}")
        return (f"{vowel}{consonant}", f"{vowel} {consonant}")


@dataclass(frozen=True, slots=True)
class VoicebankCapability:
    profile: VoicebankProfile | None
    confidence: float
    alias_coverage: float
    metadata_files: tuple[Path, ...]
    reason: str

    @property
    def supported(self) -> bool:
        return self.profile is not None and self.confidence >= 0.7


def detect_voicebank_profile(bank: Voicebank) -> VoicebankCapability:
    """Identify a supported convention from metadata and the actual alias set."""
    aliases = frozenset(unit.alias for unit in bank.units)
    presamp_files = tuple(
        path
        for path in bank.root.rglob("presamp.ini")
        if path.is_file()
    )
    if presamp_files:
        profile = _english_presamp_profile()
        coverage = _probe_coverage(profile, aliases)
        confidence = 0.9 if coverage >= 0.75 else 0.0
        reason = "presamp.ini declares an English vowel/consonant inventory"
        return VoicebankCapability(profile, confidence, coverage, presamp_files, reason)

    if any(alias.startswith("_") for alias in aliases) and "@ d" in aliases:
        profile = _english_vccv_profile()
        coverage = _probe_coverage(profile, aliases)
        return VoicebankCapability(
            profile,
            0.9 if coverage >= 0.75 else 0.0,
            coverage,
            (),
            "English VCCV alias markers detected",
        )

    if {"-I", "h@", "@d", "dh@", "@t"}.issubset(aliases):
        profile = _english_cvvc_profile()
        coverage = _probe_coverage(profile, aliases)
        return VoicebankCapability(
            profile,
            0.95 if coverage >= 0.75 else 0.0,
            coverage,
            (),
            "Aiko-style English CVVC aliases detected",
        )

    return VoicebankCapability(
        None,
        0.0,
        0.0,
        presamp_files,
        "no supported English CVVC, VCCV, or Presamp profile detected",
    )


def resolve_alias_candidates(
    phones: tuple[str, ...], bank: Voicebank, profile: VoicebankProfile
) -> tuple[AliasCandidate, ...]:
    """Return available aliases in acoustic order, preserving phone overlap."""
    symbols = tuple(profile.symbol_for(phone) for phone in phones)
    vowels = tuple(index for index, symbol in enumerate(symbols) if symbol in profile.vowels)
    return profile.candidates(symbols, vowels, frozenset(unit.alias for unit in bank.units))


def _available(
    forms: Iterable[str], indices: tuple[int, ...], aliases: frozenset[str]
) -> tuple[AliasCandidate, ...]:
    for rank, alias in enumerate(forms):
        if alias in aliases:
            return (AliasCandidate(alias, indices, rank),)
    return ()


def _deduplicate(candidates: Iterable[AliasCandidate]) -> tuple[AliasCandidate, ...]:
    seen: set[tuple[str, tuple[int, ...]]] = set()
    result: list[AliasCandidate] = []
    for candidate in candidates:
        key = (candidate.alias, candidate.phone_indices)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return tuple(result)


def _probe_coverage(profile: VoicebankProfile, aliases: frozenset[str]) -> float:
    """Score a profile against common English phone transitions, not bank size."""
    probe = [
        "aɪ", "h", "æ", "d", "ð", "æ", "t", "k", "j", "uː", "ɹ", "ɪ", "ɑː",
        "s", "ɪ", "b", "ɪ", "s", "aɪ", "d", "m", "i", "æ", "t", "ð", "ɪ",
        "s", "m", "oʊ", "m", "ə", "n", "t",
    ]
    symbols = tuple(profile.symbol_for(phone) for phone in probe)
    vowels = tuple(index for index, symbol in enumerate(symbols) if symbol in profile.vowels)
    candidates = profile.candidates(symbols, vowels, aliases)
    covered = {index for candidate in candidates for index in candidate.phone_indices}
    return len(covered) / len(probe)


def _english_cvvc_profile() -> VoicebankProfile:
    return VoicebankProfile(
        "aiko-cvvc",
        "english-cvvc",
        {
            "aɪ": "I", "æ": "@", "ɑː": "9", "ɑ": "9", "ə": "u", "ʌ": "u",
            "ɪ": "i", "i": "E", "iː": "E", "ʊ": "6", "u": "o", "uː": "o",
            # wav2vec2 may emit monophthong/lengthened variants for English
            # diphthongs (for example ``o``/``oː`` for ``oʊ`` and ``eː`` for
            # ``eɪ``). Treat them as the corresponding bank vowel so short
            # words do not disappear simply because of a label variant.
            "oʊ": "O", "o": "O", "oː": "O",
            "eɪ": "e", "eː": "e",
            "ɛ": "e", "ɜː": "3", "h": "h", "d": "d", "ɾ": "d",
            "ð": "dh", "t": "t", "k": "k", "j": "y", "ɹ": "r", "r": "r",
            "s": "s", "b": "b", "m": "m", "n": "n", "ŋ": "ng", "l": "l",
            "w": "w", "f": "f", "v": "v", "θ": "th", "ʃ": "sh", "ʒ": "zh",
            "p": "p", "ɡ": "g",
        },
        frozenset({"I", "@", "9", "u", "i", "E", "o", "O", "e", "3", "6"}),
    )


def _english_vccv_profile() -> VoicebankProfile:
    profile = _english_cvvc_profile()
    replacements = dict(profile.ipa_symbols)
    replacements.update({"ʊ": "6", "ɜː": "3"})
    return VoicebankProfile("english-vccv", "english-vccv", replacements, profile.vowels)


def _english_presamp_profile() -> VoicebankProfile:
    return VoicebankProfile(
        "english-presamp",
        "english-presamp",
        {
            "aɪ": "aI", "æ": "{", "ɑː": "A", "ɑ": "A", "ə": "@", "ʌ": "V",
            "ɪ": "i", "i": "I", "iː": "I", "ʊ": "U", "u": "u", "uː": "u",
            # See the CVVC profile above for why the short/lengthened forms
            # are accepted in addition to the canonical diphthongs.
            "oʊ": "oU", "o": "oU", "oː": "oU",
            "eɪ": "eI", "eː": "eI",
            "ɛ": "E", "ɜː": "3", "h": "h", "d": "d", "ɾ": "d",
            "ð": "D", "t": "t", "k": "k", "j": "j", "ɹ": "r", "r": "r",
            "s": "s", "b": "b", "m": "m", "n": "n", "ŋ": "N", "l": "l",
            "w": "w", "f": "f", "v": "v", "θ": "T", "ʃ": "S", "ʒ": "Z",
            "p": "p", "ɡ": "g",
        },
        frozenset({"aI", "{", "A", "@", "V", "i", "I", "U", "u", "oU", "E", "3"}),
    )


def read_presamp_metadata(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read the useful declarative parts of a ``presamp.ini`` file."""
    raw = Path(path).read_bytes()
    text = None
    for encoding in ("utf-8-sig", "cp932", "shift_jis"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"Unsupported presamp.ini encoding: {path}")
    result: dict[str, tuple[str, ...]] = {}
    current: str | None = None
    keys: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].upper()
            keys.setdefault(current, [])
        elif current in {"VOWEL", "CONSONANT", "REPLACE"} and "=" in line:
            keys[current].append(line.split("=", 1)[0].strip())
        elif current == "PRIORITY":
            keys[current].extend(part.strip() for part in line.split(",") if part.strip())
    for section in ("VOWEL", "CONSONANT", "REPLACE", "PRIORITY"):
        if section in keys:
            result[section] = tuple(keys[section])
    return result
