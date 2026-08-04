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

Milestone 8 can render that same timeline into a voicebank-derived WAV in the same run:

```powershell
synthstream-recognize human.wav voicebank/my-bank `
  --output timeline.json `
  --output-wav voicebank-output.wav
```

The renderer selects each timeline segment's real OTO unit and section, stretches it to the
detected duration, resamples voicebank recordings to one output rate, and writes actual
voicebank audio. Silence entries remain silence. The default output rate is the first loaded
voicebank unit's sample rate; `--output-sample-rate` can override it. `--pitch-ratio` applies a
single global pitch ratio while per-phone pitch transfer remains future work.

The production path uses a wav2vec2 IPA CTC model to detect phones directly from the
waveform. It does not recognize words and does not invoke a pronunciation dictionary or
G2P. A bank-specific phonemizer profile then maps the phones into aliases that actually
exist in the loaded bank. The current supported English subset is Aiko-style CVVC, English
VCCV, and English Presamp/CVVC banks with `presamp.ini`. Each alias expands into its real
OTO-derived onset, transition, and sustain sections. The transition section is centered on
the observed acoustic phone boundary, so the three sections receive independent stretch
ratios rather than merely sharing one fixed duration.

The first use downloads the approximately 1.3 GB phoneme-model weights into the Hugging
Face cache; subsequent offline runs load that cache without making a network request.

For banks without a verified profile, the earlier acoustic segmental decoder remains a
fallback and the JSON field `recognition_mode` says `acoustic-segmental`. Its output must not
be treated as semantically validated. A supported result reports a profile-specific mode
such as `wav2vec2-ipa-ctc-aiko-cvvc`, leaves `transcript` empty, and records the direct
acoustic output in `detected_phonemes` and `unmapped_phonemes`. It also reports the profile,
profile confidence, and probe alias coverage used to decide whether the bank is supported.

The timeline records each selected voicebank unit and section, start/end time, duration,
stretch ratio, silence state, costs, and diagnostics. Input WAV files may be mono or
multichannel and are resampled to the configured analysis rate when necessary.

## Streaming decoder and lookahead

Milestone 9 adds incremental decoding for the live path. Feature batches can be delivered
in callback-sized chunks while the beam and section-score cache stay alive between updates:

```python
stream = decoder.stream(lookahead_frames=3)
update = stream.push(first_feature_chunk)
for segment in update.committed_segments:
    consume_committed_segment(segment)

provisional = update.provisional_path  # may change with later chunks
final = stream.push(last_feature_chunk, final=True)
```

Committed segments trail the newest input by the configured fixed lag and must be shared by
the surviving beam hypotheses. The provisional path remains inspectable for diagnostics and
can be corrected by future evidence before commitment. `finish()` can be used instead of a
final push when no additional feature chunk remains. For live use, the convenience
`decoder.stream()` configuration uses a small candidate beam, an 800 ms maximum start
history, and a 1-second maximum section duration; these bound work as a stream grows. Pass
`candidate_limit=None`, `maximum_hypotheses=None`, `beam_threshold=None`,
`max_start_lookback_frames=None`, or `max_segment_frames=None` to opt back into exhaustive
search where latency is less important. Milestone 10 will connect this decoder to live
microphone analysis and rendering.

The optimized generic acoustic path processes the 3.4-second Aiko human-speech fixture in
approximately 1.3 seconds for decoding plus 0.2 seconds for feature extraction on the
development CPU. Exact times vary by hardware and bank size; the streaming defaults trade
some exhaustive alternatives for bounded live latency.

## Live microphone conversion

Milestone 10 connects the direct-IPA pipeline for supported English voicebanks to a duplex
microphone/output backend:

```python
from synthstream.audio import SoundDeviceDuplexBackend
from synthstream.live import LiveVoicebankEngine
from synthstream.voicebank import load_voicebank

engine = LiveVoicebankEngine(
    load_voicebank("voicebank/Kikyuune Aiko RockLoud CVVC EN"),
    SoundDeviceDuplexBackend(input_device=None, output_device=None),
)
engine.start()  # bounded audio callback plus background processing worker
try:
    ...
finally:
    engine.stop(flush=True)
```

The callback only moves fixed-size samples through bounded ring buffers. The worker analyzes
short rolling windows with the wav2vec2 direct-IPA frontend, plans aliases against the detected
voicebank profile, renders committed OTO sections, and queues output audio. For an unsupported
bank, pass `use_direct_ipa=False` to explicitly select the lower-accuracy acoustic fallback.
`start(background=False)` plus `process_available()` provides a deterministic worker-free mode
for tests and integrations that own their processing loop.
