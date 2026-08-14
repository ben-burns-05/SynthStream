"""Voicebank waveform rendering."""

from synthstream.rendering.events import (
    AliasEvent,
    allocate_alias_section_durations,
    allocate_sustain_only_durations,
)
from synthstream.rendering.overlap import BufferedOverlapComposer
from synthstream.rendering.renderer import (
    RenderResult,
    VoicebankRenderer,
)
from synthstream.rendering.scheduler import (
    RenderAppend,
    RenderSegment,
    VoicebankRenderScheduler,
    fit_audio_length,
    resample_audio,
)

__all__ = [
    "BufferedOverlapComposer",
    "AliasEvent",
    "allocate_alias_section_durations",
    "allocate_sustain_only_durations",
    "RenderResult",
    "VoicebankRenderer",
    "RenderAppend",
    "RenderSegment",
    "VoicebankRenderScheduler",
    "fit_audio_length",
    "resample_audio",
]
