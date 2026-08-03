# Milestone 9 validation

Milestone 9 makes the segmental beam decoder incremental and adds fixed-lag commitment.

## Streaming behavior

`SegmentalBeamDecoder.stream(lookahead_frames=N)` returns a
`StreamingSegmentalBeamDecoder`. Each `push(FeatureBatch)` call appends new analysis frames,
extends only the newly reachable dynamic-programming endpoints, and retains the existing
beam and score cache. The update exposes:

- `provisional_path`: the current best path, which may change as future frames arrive;
- `committed_segments`: newly emitted sections that are safe to synthesize;
- `committed_path`: the complete prefix emitted so far;
- `decode_result`: current alternatives and beam diagnostics.

Before finalization, a segment is committed only when all surviving hypotheses share it and its
end lies at least `lookahead_frames` frames behind the newest input. `push(..., final=True)` or
`finish()` commits the winning path through the end of the stream.

## Regression coverage

The decoder tests verify that:

1. Incremental processing produces the same final path as batch decoding.
2. Commitments respect the fixed lookahead lag.
3. An initially attractive path remains uncommitted while a later section can select a
   different path, then the corrected path is committed at finalization.
4. Invalid lookahead settings are rejected.
