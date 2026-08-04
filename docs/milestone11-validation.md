# Milestone 11 validation

Milestone 11 applies the overlap values already present in each OTO entry to
both offline WAV synthesis and the live engine.

## Overlap behavior

Adjacent sections from the same OTO unit remain contiguous. When a new alias
unit starts, its `overlap_ms` is used as the crossfade length. The overlap is
warped by the new unit's onset stretch ratio, so a stretched onset also gets a
proportionally longer transition. Silence resets the overlap context.

Offline synthesis keeps the complete timeline mutable until the WAV is
assembled, preserving the requested input duration while mixing each new
section into the previous section's tail.

The live engine uses a 120 ms staging window. Audio older than that window is
released to the output ring buffer and is immutable; a newly committed section
can still crossfade into the recent tail. `flush()` releases the remaining
staged audio.

## Verification

The real Aiko phonetic timeline test reports more than ten OTO overlap
transitions and verifies non-silent output. Unit tests cover crossfading,
nominal duration preservation, immutable-prefix release, and the existing live
fake-backend and real-bank paths.
