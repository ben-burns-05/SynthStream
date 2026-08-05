# Milestone 12 validation

Milestone 12 adds a usable desktop control surface around
`LiveVoicebankEngine`.

## GUI controls

`MainWindow` now provides:

- voicebank-folder selection and load-state/unit-count reporting;
- input and output device selectors, with safe system-default fallbacks when
  no hardware is available;
- Start and Stop controls wired to the production live engine;
- live diagnostics for input/output buffer occupancy, committed sections,
  rendered samples, processing time, and transport/worker errors.

The GUI uses the same `LiveVoicebankEngine` as programmatic integrations. The
engine remains responsible for audio callbacks, analysis, decoding, rendering,
and overlap staging; a Qt timer only refreshes diagnostics, so model work does
not run on the GUI thread. Direct-IPA model assets are prepared on a startup
worker before the transport opens, preventing cold-start model download time
from being reported as a stream of output underflows.

`MainWindow` accepts an already loaded `Voicebank`, a fake backend, a backend
factory, and a device provider. These injection points make deterministic GUI
end-to-end tests possible without requiring physical audio hardware.

## Verification

The GUI test launches the window, loads a test voicebank, starts the real
acoustic live engine on `FakeDuplexAudioBackend`, feeds human-like audio,
stops with a flush, and verifies committed sections plus non-silent audio in
the output transport. The original application launch and QApplication reuse
smoke tests remain covered.
