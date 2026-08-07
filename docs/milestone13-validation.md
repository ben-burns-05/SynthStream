# Milestone 13 validation

Milestone 13 closes the application-level test loop around the production
pipeline.

## Covered surfaces

- analysis and matching cover complete bank indexing, cache reuse, full-bank
  ranking, duration constraints, and component score behavior;
- decoding covers ordered sections, silence, future-evidence beam recovery,
  streaming commitment, and invalid configurations;
- offline recognition and synthesis cover real WAV input, timeline JSON, WAV
  output, silence placement, real Aiko aliases, and OTO overlap transitions;
- audio and live-engine tests cover fake duplex transport, bounded buffers,
  output underflow reporting, acoustic fallback conversion, direct real-bank
  conversion, background workers, and idempotent finalization;
- GUI tests cover application launch, voicebank gating, device-discovery
  fallback, real engine start/stop, matching, rendering, output transport, and
  status updates. When the local Aiko bank is installed, a second GUI test
  exercises the default direct-IPA path with the real human fixture and model.

The GUI end-to-end test deliberately uses the same `LiveVoicebankEngine` and
`FakeDuplexAudioBackend` as non-GUI integrations, so it does not substitute a
mock conversion path for the production matcher or renderer.

## Verification result

The complete suite passes with 119 tests and 88% branch-aware coverage on the
development environment. Ruff and mypy also pass. Tests that require optional
development voicebanks skip cleanly when those local banks are unavailable.
