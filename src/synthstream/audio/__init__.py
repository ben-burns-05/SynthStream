"""Audio output and realtime transport abstractions."""

from synthstream.audio.buffer import AudioRingBuffer
from synthstream.audio.output import AudioSink, SoundDeviceSink, WavFileSink
from synthstream.audio.stream import (
    DuplexAudioBackend,
    FakeDuplexAudioBackend,
    RealtimeAudioStream,
    SoundDeviceDuplexBackend,
    StreamDiagnosticEvent,
    StreamStatistics,
)

__all__ = [
    "AudioRingBuffer",
    "AudioSink",
    "DuplexAudioBackend",
    "FakeDuplexAudioBackend",
    "RealtimeAudioStream",
    "SoundDeviceDuplexBackend",
    "SoundDeviceSink",
    "StreamDiagnosticEvent",
    "StreamStatistics",
    "WavFileSink",
]
