# SynthStream

SynthStream is an early-stage desktop application for converting live human speech into audio assembled from an UTAU/VOCALOID-style voicebank.

## Development setup

Python 3.11 or newer is required. From a clean virtual environment:

```powershell
python -m pip install -e ".[dev]"
```

Run the application and quality checks with:

```powershell
synthstream
pytest
ruff check .
mypy
```

The immutable initial requirements are in `docs/initial_spec.md`. Implementation decisions and evolving requirements belong in `docs/spec.md`.

## Loading a voicebank

Milestone 1 provides a recursive UTAU `oto.ini` loader:

```python
from synthstream.voicebank import load_voicebank

bank = load_voicebank("voicebank/my-bank")
print(len(bank.units))
```

Parsed metadata is cached inside the selected bank's ignored `.synthstream-cache/` directory. The cache is automatically invalidated when an `oto.ini` file or referenced WAV changes.

## Rendering a voicebank unit

Milestone 2 can render a loaded unit at an independently selected duration and pitch, then route the result through a file or device sink:

```python
from synthstream.audio import WavFileSink
from synthstream.rendering import VoicebankRenderer
from synthstream.voicebank import load_voicebank

unit = load_voicebank("voicebank/my-bank").units[0]
result = VoicebankRenderer().render_unit(
    unit,
    duration_seconds=0.75,
    pitch_ratio=1.25,
)
result.send_to(WavFileSink("rendered.wav"))
```

## Realtime audio transport

Milestone 3 keeps the physical audio callback limited to bounded sample movement. Processing workers read microphone samples and enqueue synthesized output independently:

```python
from synthstream.audio import RealtimeAudioStream, SoundDeviceDuplexBackend

backend = SoundDeviceDuplexBackend(input_device=None, output_device=None)
stream = RealtimeAudioStream(backend, sample_rate=16_000, block_size=320)
stream.start()

human_audio = stream.read_input(320)
stream.write_output(synthesized_audio)

stream.stop()
```

`FakeDuplexAudioBackend` drives the same production transport in deterministic callback-sized blocks for automated tests.

## Human audio analysis

Milestone 4 extracts configurable frame features for matching while keeping absolute F0 separate for later prosody transfer:

```python
from synthstream.analysis import AnalysisConfig, FeatureExtractor

extractor = FeatureExtractor(
    AnalysisConfig(sample_rate=16_000, window_ms=25, hop_ms=10, mel_bands=40)
)
features = extractor.analyze(human_audio)

print(features.recognition_features.shape)
print(features.f0_hz, features.voiced)
```

The recognition matrix contains normalized log-mel spectra, spectral deltas, energy and delta energy, spectral flux, flatness, and periodicity. F0 and RMS energy remain available as separate synthesis/prosody signals.

## Voicebank section matching

Milestone 5 precomputes the same recognition features for every loaded voicebank section and uniformly warps section trajectories when scoring human intervals:

```python
from synthstream.analysis import FeatureExtractor
from synthstream.matching import SectionFeatureIndex, SectionMatcher
from synthstream.voicebank import load_voicebank

extractor = FeatureExtractor()
bank = load_voicebank("voicebank/my-bank")
index = SectionFeatureIndex.build(bank, extractor)
matcher = SectionMatcher(index)

human = extractor.analyze(human_audio)
ranked = matcher.match_interval(human, start_frame=0, end_frame=human.frame_count)
print(ranked[0].template.alias, ranked[0].total_cost)
```

Each result retains separate spectral, delta, periodicity, flatness, flux, energy, and duration costs. The feature index caches trajectories for unchanged banks and supports vectorized onset shortlisting while keeping the complete vocabulary available.

## Segmental decoding

Milestone 6 jointly searches section identity and duration through the voicebank graph, with silence represented inside the same beam:

```python
from synthstream.decoding import SegmentalBeamDecoder

decoder = SegmentalBeamDecoder(matcher)
result = decoder.decode(human)

for segment in result.best_path.segments:
    print(segment.alias, segment.section_kind, segment.start_frame, segment.end_frame)
```

The result retains alternative final paths, per-segment stretch and cost details, and beam diagnostics. Within-unit section order is enforced, switching units is penalized, long silence is supported, and multiple early interpretations can remain alive until later sections resolve them.

## Offline WAV recognition

Milestone 7 exports a machine-readable real-voicebank timeline from prerecorded human speech:

```powershell
synthstream-recognize human.wav voicebank/my-bank --output timeline.json
```

For Kikyuune Aiko RockLoud CVVC EN, the production path uses a wav2vec2 IPA CTC
model to detect phones directly from the waveform. It does not recognize words and does
not invoke a pronunciation dictionary or G2P. The detected IPA phones are mapped into
aliases that actually exist in the loaded bank. Each alias expands into its real OTO-derived
onset, transition, and sustain sections. The transition section is centered on the observed
acoustic phone boundary, so the three sections receive independent stretch ratios rather
than merely sharing one fixed duration.

The first use downloads the approximately 1.3 GB phoneme-model weights into the Hugging
Face cache; subsequent offline runs load that cache without making a network request.

For banks without a verified phoneme map, the earlier acoustic segmental decoder remains a
fallback and the JSON field `recognition_mode` says `acoustic-segmental`. Its output must not be
treated as semantically validated. A supported Aiko result instead reports
`wav2vec2-ipa-ctc-aiko-cvvc`, leaves `transcript` empty, and records the direct acoustic
output in `detected_phonemes` and `unmapped_phonemes`.

The timeline records each selected voicebank unit and section, start/end time, duration,
stretch ratio, silence state, costs, and diagnostics. Input WAV files may be mono or
multichannel and are resampled to the configured analysis rate when necessary.
