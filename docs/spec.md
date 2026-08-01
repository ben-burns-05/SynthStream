# Project Specification

# Live Human Speech → UTAU/VOCALOID-Style Voicebank Conversion

## 1. Overview

This project is a real-time speech conversion application.

It takes a live stream of human speech, determines which parts of a loaded UTAU/VOCALOID-style voicebank best correspond to that speech, and produces a new live audio stream synthesized from the selected voicebank recordings.

The intended user experience is:

```text
select voicebank
select microphone/input device
select output device
press Start
speak
hear the same speech reconstructed using the selected voicebank
```

The application should have a basic desktop GUI and should also support offline WAV-based testing and conversion.

The main technical challenge is not general speech recognition. It is aligning variable-duration human speech against variable-duration subsections of a voicebank while preserving the deliberately sampled/concatenative UTAU/VOCALOID character.

---

# 2. Project Setup

Use Python as the primary implementation language.

Create a clean reproducible Python project.

The initial repository should contain at least:

```text
src/
tests/
voicebank/
docs/
```

The internal structure below `src/` and `tests/` is intentionally flexible and can evolve with the implementation.

Do not spend time designing a large speculative package hierarchy before it is useful.

Use a modern `pyproject.toml`-based environment with reproducible dependency management.

A suitable initial dependency set is:

```text
numpy
scipy
torch
torchaudio
sounddevice
soundfile
PySide6

pytest
pytest-qt
pytest-cov
ruff
mypy
```

Additional dependencies may be introduced where useful.

The project should support:

```text
installing from a clean environment
running the application
running tests
running ruff
running mypy
```

## Project specifications

Create:

```text
docs/initial_spec.md
docs/spec.md
```

`docs/initial_spec.md` should contain the initial project specification and then remain unchanged.

`docs/spec.md` should initially contain the same contents, but it is the working project specification and may be updated as implementation details, experiments, performance results, or requirements evolve.

The `voicebank/` directory is intended for local development/testing voicebanks. Voicebank audio does not necessarily need to be committed to version control.

---

# 3. End-to-End System

The main runtime pipeline is:

```text
human audio stream
        ↓
audio analysis / feature extraction
        ↓
segment-level voicebank matching
        ↓
segmental beam/Viterbi decoder
        ↓
committed voicebank section timeline
        ↓
voicebank waveform renderer
        ↓
time stretching / pitch / overlap
        ↓
output audio stream
```

Recognition and synthesis are separate stages, but both belong to the same application.

The recognizer determines what voicebank material should be used and its timing.

The renderer turns those decisions into actual output audio using the original voicebank recordings.

---

# 4. Voicebank Loading

The application should load an entire selected voicebank.

The first target is UTAU-style voicebanks using `oto.ini`-style metadata and referenced WAV files.

The loader should convert that information into a generic internal representation suitable for both matching and synthesis.

The production system must not be built around a hardcoded subset of phonemes.

If the voicebank contains hundreds or thousands of usable entries, those entries should remain available to the matching system.

Small voicebanks are useful for tests, but they are test fixtures rather than a limitation of the actual application.

Voicebank preprocessing should be cached so that unchanged banks do not need to be fully analysed every time the program launches.

---

# 5. Voicebank Sections

The matcher should operate on **subsections of voicebank entries**, not on whole phonemes as indivisible templates.

A voicebank unit may contain sections such as:

```text
consonant
preutterance
transition
constant
vowel
release
```

The exact set of section types can depend on the voicebank and may evolve as the implementation becomes better understood.

Different units do not need to have identical section layouts.

For example:

```text
/ka/

consonant
transition
constant
vowel
```

and:

```text
/a/

attack
constant
vowel
```

may both be valid.

The decoder works with these sections directly.

A phoneme or voicebank unit is identified by the path the decoder takes through its sections.

For example:

```text
KA_CONSONANT
    ↓
KA_TRANSITION
    ↓
KA_CONSTANT
    ↓
KA_VOWEL
```

is effectively the `/ka/` interpretation.

There should not be a separate prerequisite step that first chooses `/ka/` and only afterwards determines its timing.

---

# 6. Why Section Matching Is Used

The central timing problem is that neither phoneme identity nor phoneme duration is known in advance.

The system cannot first decide:

```text
this is /ka/
```

and then determine its stretch factor, because the stretch factor affects how well `/ka/` matches.

Likewise, it cannot determine the correct stretch factor without already knowing what template is being stretched.

The decoder therefore searches both together.

The fundamental candidate is:

```text
voicebank section
+
human-audio start
+
human-audio end
```

The human interval defines the candidate duration.

The candidate voicebank section is then uniformly stretched/compressed to that duration and scored.

---

# 7. Uniform Section Warping

Each section uses one uniform time-scale ratio.

If a voicebank section contains 12 analysis frames and the candidate human section contains 20 frames, the voicebank feature trajectory is linearly resampled:

```text
12 frames → 20 frames
```

and compared against the 20 human frames.

The warp rate stays constant through the section.

Example:

```text
voicebank vowel duration: 100 ms
human vowel duration:     170 ms

stretch ratio: 1.70×
```

The entire vowel uses approximately that 1.70× mapping.

Neighbouring sections may have different ratios:

```text
consonant    0.95×
transition   1.20×
constant     1.05×
vowel        1.75×
```

This is intentional.

Unrestricted DTW is not used for matching because it can continuously alter the warp rate within the section and destroy the more rigid UTAU/VOCALOID timing character.

---

# 8. Human Audio Analysis

Human input should be converted into short frame-based feature vectors.

A reasonable initial analysis setup is approximately:

```text
hop:       5–10 ms
window:   20–30 ms
```

These values should remain configurable.

The initial feature set should include:

```text
log-mel spectral features
delta / spectral-change features
normalized energy
delta energy
spectral flux
spectral flatness
periodicity / voicing confidence
```

Pitch/F0 should also be estimated, but mainly for synthesis rather than phoneme identity.

The exact feature representation can later be improved if experiments show that a learned embedding works better.

---

# 9. Speaker Independence

Human speech and voicebank speech come from different speakers.

The matcher should therefore avoid relying too strongly on:

```text
absolute pitch
absolute loudness
speaker-specific timbre
microphone response
```

Useful information is instead:

```text
spectral shape
spectral movement
periodic vs noisy behaviour
onsets
transitions
vowel/consonant structure
silence
```

Feature normalization should be used to reduce speaker and recording differences.

Possible approaches include:

```text
log amplitude scaling
mean normalization
variance normalization
local normalization
```

The exact normalization method can be tuned experimentally.

---

# 10. Voicebank Feature Precomputation

Voicebank analysis should normally happen before realtime matching.

For every usable voicebank section, precompute and cache the same type of recognition features used for human speech.

A section should conceptually retain:

```text
voicebank unit identity
section identity/type
source WAV
source waveform boundaries
nominal duration
allowed duration/stretch range
duration-model parameters
precomputed feature trajectory
rendering metadata
```

At runtime the expensive spectral analysis of the voicebank should not be repeated for every decoder candidate.

---

# 11. Section Acoustic Score

For a candidate section:

```text
voicebank section S
human frames [start:end]
```

let:

```text
d = end - start
```

The cached voicebank feature trajectory is uniformly resampled to `d` frames.

The human and voicebank trajectories are then compared.

The initial acoustic score may combine terms such as:

```text
mel-spectrum distance
delta-spectrum distance
periodicity difference
spectral-flatness difference
spectral-flux difference
normalized-energy difference
```

Conceptually:

```text
AcousticCost =
    w_mel       * mel_cost
  + w_delta     * delta_cost
  + w_periodic  * periodicity_cost
  + w_flatness  * flatness_cost
  + w_flux      * flux_cost
  + w_energy    * energy_cost
```

The individual components should remain inspectable during development.

Different section types may eventually use different feature weights.

For example, a fricative may care more about spectral/noise characteristics while a stable vowel may care more about spectral shape and periodicity.

---

# 12. Duration Score

Acoustic similarity alone is insufficient.

A candidate should also be penalized when it requires an implausible duration transformation.

For:

```text
nominal duration = L
candidate duration = d
stretch = d / L
```

a useful initial penalty is:

```text
DurationCost =
    lambda * log(stretch)^2
```

The exact formula may be adjusted during development.

Different section types should be able to use different duration constraints.

For example:

```text
short consonant burst:
relatively narrow allowed range

vowel:
much wider allowed range
```

Each section or section type should therefore be able to specify:

```text
minimum duration/stretch
maximum duration/stretch
duration penalty strength
```

---

# 13. Segmental Beam/Viterbi Decoder

The intended recognition algorithm is a **segmental beam-search / Viterbi-style decoder operating on voicebank sections**.

The decoder jointly searches:

```text
which section is active
when it started
when it ends
how much it stretches
which section comes next
which unit/phoneme path is most plausible
whether the input is silence
```

A decoder hypothesis therefore represents a partial path through voicebank sections.

Conceptually it contains information such as:

```text
current section
section start frame
accumulated path score
previous path / backpointer
```

Multiple hypotheses coexist.

There is not one global `current_phoneme`.

---

# 14. Segment Completion

For an active section beginning at frame `s`, as new frames arrive the decoder can consider:

```text
continue this section
```

or:

```text
finish this section at the current frame
```

If the section finishes at frame `t`, the decoder evaluates:

```text
AcousticCost(section, audio[s:t])
+
DurationCost(section, t-s)
+
TransitionCost(previous_section, next_section)
```

and spawns valid next states.

The exact optimization implementation can vary, but the resulting search should be equivalent to this segment-level reasoning.

---

# 15. Transition Scoring

The transition model represents voicebank/phoneme structure.

Transitions between sequential sections of the same unit are normally cheap.

Example:

```text
KA_CONSONANT → KA_TRANSITION
KA_TRANSITION → KA_CONSTANT
KA_CONSTANT → KA_VOWEL
```

This makes continuing the current phoneme/unit preferable when the audio still fits it.

Starting a different unit should generally have some additional cost.

This provides useful inertia without making the current interpretation impossible to overturn.

Invalid section transitions should simply not be part of the graph where practical.

---

# 16. New Phoneme Starts

A high likelihood that another phoneme is beginning should not immediately force a switch.

Instead it should make new transition branches competitive inside the beam.

Example:

```text
current path:
A_VOWEL
```

may produce:

```text
continue A_VOWEL

A → B_START

A → C_START

A → SILENCE
```

These alternatives remain in competition.

A later section of B may confirm that B was correct, or later evidence may eliminate that hypothesis.

---

# 17. Future Evidence and Lookahead

Future sections must be able to influence the interpretation of earlier sections.

Example:

```text
early audio:

TA path   score 10
KA path   score 11
```

TA currently appears better.

Later:

```text
TA path   score 30
KA path   score 18
```

The KA continuation fits much better.

The decoder should then retain and ultimately select the earlier KA interpretation.

This means the system should not immediately commit the locally best phoneme.

A beam of plausible paths should remain alive.

A short amount of lookahead/delayed commitment is therefore expected.

Initially, fixed-lag commitment is a reasonable implementation:

```text
decoder processes current audio
committed result trails decoder by N milliseconds
```

More adaptive shared-prefix commitment may be explored later.

---

# 18. Backtracking / Commitment

Decoder paths should retain backpointers or equivalent history so the winning path can recover:

```text
selected voicebank unit
selected sections
section starts
section ends
stretch factors
transitions
silence regions
```

The system should distinguish between:

```text
best provisional path
```

and:

```text
committed path safe to synthesize
```

The provisional path may change as new audio arrives.

---

# 19. Silence

Silence should participate in the same decoder.

The decoder should be capable of choosing among:

```text
continue current vowel
begin another phoneme
enter silence
remain in silence
```

A silence score can initially use features such as:

```text
low energy
low spectral activity
low or undefined periodicity
```

Silence should support long durations.

It should not be implemented purely as a preprocessing switch that disables normal decoding.

---

# 20. Full Voicebank Candidate Search

When a new unit may be starting, candidates should come from the complete loaded voicebank.

For performance, a coarse-to-fine strategy is appropriate.

For example:

```text
current human onset features
        ↓
cheap/vectorized comparison against all possible start sections
        ↓
rank candidates
        ↓
perform detailed segment scoring on the most promising candidates
```

This is only a search optimization.

It must still be possible for any appropriate voicebank entry to be selected.

The exact indexing method can be chosen after profiling.

---

# 21. Optional Learned Matching Features

If conventional log-mel/spectral features prove insufficiently speaker-independent, a small learned acoustic representation may be explored.

For example:

```text
log-mel frames
    ↓
small CNN / TCN / GRU
    ↓
phonetic embedding
```

The same encoder would process both human speech and voicebank recordings.

Metric-learning approaches such as contrastive or triplet training may be useful.

This is an extension to the core matching architecture, not a replacement for the segmental decoder.

---

# 22. Pitch and Prosody Analysis

Human pitch and prosody should be tracked separately from phonetic matching.

Useful values include:

```text
F0
voicing confidence
energy/prosody envelope
```

A lightweight pitch estimator such as a YIN-style approach is an appropriate starting point.

The previous implementation used `pyworld` and it was too slow, so WORLD should not be relied on for the realtime path.

Absolute human F0 should generally not decide which phoneme was spoken.

Instead, F0 is mainly used after a voicebank unit has been selected.

---

# 23. Voicebank Rendering

The decoder output must drive an actual voicebank renderer.

For each committed section:

```text
select corresponding original voicebank WAV region
        ↓
change its duration to the chosen section duration
        ↓
apply target pitch/intonation when appropriate
        ↓
apply broad human energy/prosody
        ↓
combine with neighbouring rendered sections
        ↓
write into output timeline
```

The output waveform must come from actual voicebank recordings.

Recognition features are only used for matching.

They are not the output representation.

---

# 24. Time and Pitch Transformation

Time stretching and pitch shifting should be conceptually separate.

A section may need:

```text
time stretch: 1.60×
pitch ratio:  0.85×
```

These should not be coupled through naive resampling.

The exact rendering DSP can evolve.

Possible initial approaches include:

```text
PSOLA/TD-PSOLA-like rendering for voiced material

OLA/WSOLA-like stretching for unvoiced/noisy material
```

or another suitable implementation/library.

The renderer should be designed so the underlying DSP approach can be improved without changing the recognition architecture.

---

# 25. Human Prosody Transfer

The rendered voicebank speech should approximately follow the human speaker's:

```text
pitch contour
intonation
basic loudness/emphasis
```

For voiced sections, map the human F0 trajectory onto the synthesized section timeline.

For unvoiced sections, pitch is not meaningful and should not be forced.

The human energy contour may be normalized and used as a broad gain envelope.

---

# 26. Voicebank Overlap

Voicebank overlap is a real overlap in synthesized audio.

The matcher does not initially need to acoustically detect or score overlap separately.

Instead:

```text
decoder commits transition A → B
        ↓
voicebank metadata provides B's overlap timing
        ↓
renderer applies B overlapping the end of A
```

So the recognition timeline might be:

```text
A ====================|================ B
```

while the rendered waveform is closer to:

```text
A =========================
                 B =========================
                 ^^^^^^^
                 overlap
```

---

# 27. Warped Overlap

Overlap itself can stretch or compress.

A reasonable initial approach is to derive its stretch from the early/onset section of the new unit.

Example:

```text
nominal B overlap = 30 ms
B onset stretch = 1.4×

rendered overlap ≈ 42 ms
```

This mapping can later be adjusted if listening tests suggest a better rule.

The initial system does not need to perform a separate acoustic search for optimal overlap duration.

---

# 28. Retroactive Overlap

The decoder may only become confident about `A → B` after some future audio has already arrived.

The beginning of B's overlap may therefore belong slightly before the point where the transition becomes committed.

The renderer should maintain a short mutable staging buffer containing synthesized audio that has not yet been sent to the physical output device.

When `A → B` commits:

```text
calculate warped overlap
        ↓
render beginning of B into the recent buffered past
        ↓
mix it with A
        ↓
eventually release that audio to the output device
```

Audio already played cannot be modified.

This is one reason the system naturally has some output latency.

---

# 29. Live Audio Processing

The realtime audio callback should remain lightweight.

Its primary responsibilities are:

```text
input samples → input buffer

output buffer → audio device
```

Feature extraction, decoding, rendering and GUI updates should run outside the strict callback.

The exact concurrency model is an implementation decision.

Important requirements are:

```text
GUI remains responsive
audio callback does not block on heavy processing
buffers remain bounded
worker failures are visible
Start/Stop works cleanly
```

---

# 30. GUI

The application should provide a basic desktop GUI.

The exact design/layout is flexible.

At minimum it should allow:

```text
select voicebank
select audio input device
select audio output device
Start
Stop
```

Useful runtime information includes:

```text
voicebank load state
number of loaded voicebank units
input level
output level
estimated human F0
provisional matched unit
committed matched unit
current section
stretch ratio
beam size
estimated latency
audio underruns/errors
```

The GUI should control the same production engine used elsewhere in the project.

---

# 31. Offline Human WAV → Phoneme/Voicebank Timeline

The project should provide an offline path for analysing a prerecorded human WAV.

It should use the same production:

```text
feature extraction
section scoring
voicebank candidate search
segmental beam decoder
duration model
transition model
traceback
```

as the live application.

The result should include:

```text
voicebank unit/alias
section
start time
end time
duration
stretch ratio
silence
```

and should be exportable in a machine-readable format such as JSON.

This is an important development/debugging interface.

---

# 32. Offline Human WAV → Voicebank Audio

The project must also support:

```text
human.wav
        ↓
production matcher
        ↓
voicebank unit/section timeline
        ↓
production renderer
        ↓
voicebank_output.wav
```

One run should be able to produce both:

```text
recognized timeline
synthesized WAV
```

This gives a repeatable way to evaluate both recognition and synthesis without realtime audio hardware.

---

# 33. Testing Requirements

Testing is a required part of the project.

Tests should cover low-level components where useful, but several higher-level workflows are especially important.

## Phoneme/section matching tests

There must be tests that take actual human speech WAV files and run:

```text
human WAV
→ production audio analysis
→ production voicebank matching
→ production decoder
→ phoneme/unit/section timeline
```

These should verify things such as:

```text
major selected units
section order
approximate boundaries
stretch ratios
silence placement
```

Timing comparisons should use sensible tolerances.

Regression/golden fixtures are useful for tracking matching quality over time.

## Future-evidence tests

There should be explicit tests where:

```text
candidate A is initially best
candidate B remains in the beam
later audio makes B clearly better
final committed result is B
```

This tests the core lookahead/beam behaviour.

## Full-voicebank tests

There should be tests showing that valid entries throughout a complete or realistically large voicebank remain selectable.

This protects against accidental search truncation.

## Renderer tests

The renderer should be testable independently with known:

```text
voicebank unit
section timing
pitch
prosody
overlap
```

and should produce valid voicebank-derived audio.

## Human WAV → voicebank WAV tests

This is a required end-to-end backend test.

It should run:

```text
human speech WAV
        ↓
matching
        ↓
segmental decoding
        ↓
voicebank rendering
        ↓
synthesized WAV
```

and capture both the recognised timeline and output waveform.

The test should verify that:

```text
matching produces a plausible sequence
output duration is sensible
voicebank-derived audio is produced
speech regions are non-silent
silence behaves appropriately
waveform contains finite/bounded samples
```

## Live-engine tests

Use fake audio input/output where physical devices are not suitable for CI.

Feed a human WAV into the fake microphone in realtime-sized blocks and capture the generated output stream.

The production live engine should be exercised rather than a special test-only converter.

## GUI end-to-end tests

GUI end-to-end testing is required.

At least one automated test should perform approximately:

```text
launch GUI
        ↓
load test voicebank
        ↓
select fake input/output
        ↓
press Start
        ↓
feed human WAV through fake microphone
        ↓
run real matcher/decoder
        ↓
run real renderer
        ↓
capture output stream
        ↓
press Stop
```

The test should verify that:

```text
GUI remains responsive
voicebank loads
conversion starts
human audio is processed
matching results are produced
voicebank audio is synthesized
output audio reaches the fake device
status information updates
Stop works cleanly
```

Checking that buttons merely exist is not sufficient for this end-to-end test.

---

# 34. Offline / Live Consistency

The same human WAV processed through:

```text
offline conversion
```

and:

```text
fake realtime input
```

should produce broadly similar:

```text
voicebank unit sequence
section sequence
timing
synthesized speech content
```

They do not need to be sample-identical.

The purpose is to make sure offline debugging actually represents the production realtime system.

---

# 35. Debugging and Observability

During development, important matching decisions should be inspectable.

Useful information includes:

```text
top beam hypotheses
current candidate sections
candidate durations
acoustic score
duration score
transition score
total score
committed path
```

It should also be possible to optionally save from a live run:

```text
input human WAV
output synthesized WAV
decoder timeline/events
```

This makes realtime failures reproducible offline.

---

# 36. Configuration

Experimental parameters should be configurable rather than scattered as unexplained constants.

Examples include:

```text
sample rate
analysis window
hop size
mel-band count
feature weights
normalization settings
section duration constraints
duration penalties
transition/switching costs
silence parameters
beam threshold
maximum active hypotheses
candidate-retrieval count
lookahead
pitch smoothing
overlap scaling
output gain
buffer sizes
```

Not all of these need to appear in the GUI.

---

# 37. Performance Approach

Initial development should prioritize a correct observable end-to-end system.

Likely performance techniques include:

```text
precompute voicebank features
cache voicebank preprocessing
analyse human frames once
vectorize broad candidate search
perform expensive section scoring only for plausible candidates
beam pruning
finite section-duration limits
segment-score caching
batched tensor operations
waveform caching/prefetching
```

If profiling later shows that a component needs native acceleration, selected hotspots can move to C/C++/Rust or an optimized DSP library without changing the overall architecture.

The production vocabulary should not be reduced merely to make the program faster.

---

# 38. Development Milestones

## Milestone 0 — Project Setup

Create the project directory and initial:

```text
src/
tests/
voicebank/
docs/
```

Set up the reproducible Python environment, `pyproject.toml`, dependencies, `pytest`, `ruff`, and `mypy`.

Create:

```text
docs/initial_spec.md
docs/spec.md
```

with this specification.

Afterward:

```text
initial_spec.md
```

remains unchanged, while:

```text
spec.md
```

may evolve with the project.

Create enough application code to verify that the environment installs correctly, tests run, and a minimal GUI/application entry point launches.

## Milestone 1 — Voicebank Loading

Load a complete voicebank, parse its timing/audio metadata, create internal units/sections, and begin caching preprocessing results.

## Milestone 2 — Basic Voicebank Rendering

Given a manually selected voicebank unit, duration and pitch, generate actual transformed voicebank audio.

Allow this audio to be saved and sent through the output abstraction.

## Milestone 3 — Realtime Audio Infrastructure

Implement audio input/output streaming, buffering, and fake-device support for tests.

## Milestone 4 — Human Audio Analysis

Implement the initial spectral features, periodicity/voicing, F0 and prosody analysis.

## Milestone 5 — Section Matching

Implement uniformly warped section-feature comparison and full-bank candidate search.

## Milestone 6 — Segmental Beam Decoder

Implement joint section identity/duration search, section transitions, phoneme continuation preference, silence, beam pruning, and path history.

## Milestone 7 — Offline WAV → Phoneme/Unit Timeline

Run real human WAV recordings through the production matcher and decoder.

Add matching/regression tests.

## Milestone 7.5 — Voicebank Phonemizer Profiles

Separate direct acoustic IPA recognition from voicebank-format alias resolution. Detect and
validate a supported profile against the complete loaded alias inventory, read singer-local
`presamp.ini` metadata where present, resolve CVVC/VCCV/Presamp fallback combinations, and
report profile confidence and alias coverage. The initial supported subset is English Aiko-style
CVVC, English VCCV, and English Presamp/CVVC. Unsupported or ambiguous banks must remain
explicitly marked instead of receiving fabricated semantic aliases.

## Milestone 8 — Offline WAV → Voicebank Synthesis

Connect the decoder to the renderer so a human WAV produces both:

```text
matched timeline
voicebank-synthesized WAV
```

Add required end-to-end backend tests.

## Milestone 9 — Streaming Decoder and Lookahead

Make decoding incremental and implement delayed commitment so future sections can correct earlier ambiguous decisions.

## Milestone 10 — Live Human Speech → Live Voicebank Audio

Connect microphone input through the complete matcher/decoder/renderer to the selected output device.

## Milestone 11 — Voicebank Overlap

Implement real overlapping synthesis, warped overlap timing, and buffered retroactive placement.

## Milestone 12 — GUI Integration

Finish the basic usable GUI around the production engine.

## Milestone 13 — Full Test Coverage

Ensure matching, WAV synthesis, live-engine, and GUI end-to-end tests cover the complete application.

## Milestone 14 — Profiling and Quality Improvement

Evaluate realistic voicebanks and speakers.

Profile actual bottlenecks and improve:

```text
matching accuracy
section timing
speaker independence
pitch transfer
render quality
latency
CPU/GPU usage
```

Update `docs/spec.md` as the implementation evolves.

---

# 39. Project Success Criteria

A successful initial version should allow a user to:

1. Launch the application.
2. Select a complete UTAU-style voicebank.
3. Select an input audio device.
4. Select an output audio device.
5. Start conversion.
6. Speak normally.
7. Have the program continuously match the speech against the loaded voicebank.
8. Have ambiguous phoneme choices remain provisional until later audio resolves them.
9. Have selected sections uniformly stretched/compressed to follow human timing.
10. Have human pitch/prosody transferred onto the selected voicebank recordings.
11. Have voicebank overlap applied between selected units.
12. Hear the resulting voicebank-derived speech through the selected output device.
13. Stop conversion cleanly.

The repository should also support:

```text
human WAV → phoneme/unit/section timeline
```

and:

```text
human WAV → synthesized voicebank WAV
```

using the same production matching and rendering code.

Automated tests should include:

```text
phoneme/section matching
future-evidence beam behaviour
full-voicebank search
voicebank rendering
human WAV → voicebank WAV
live fake-device conversion
GUI end-to-end conversion
```
