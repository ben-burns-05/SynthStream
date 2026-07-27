import numpy as np
import pytest

from synthstream.audio import (
    FakeDuplexAudioBackend,
    RealtimeAudioStream,
    SoundDeviceDuplexBackend,
)
from synthstream.audio import stream as stream_module


def test_fake_duplex_moves_realtime_blocks_in_both_directions() -> None:
    backend = FakeDuplexAudioBackend()
    stream = RealtimeAudioStream(
        backend, sample_rate=1_000, block_size=4, buffer_duration_seconds=0.02
    )
    expected_output = np.linspace(-0.5, 0.5, 10, dtype=np.float32)
    microphone = np.arange(10, dtype=np.float32)
    stream.write_output(expected_output)

    stream.start()
    returned_output = backend.feed(microphone)
    stream.stop()

    assert np.array_equal(stream.read_input(20), microphone)
    assert np.allclose(returned_output, expected_output)
    assert np.array_equal(backend.take_captured_output(), expected_output)
    assert not stream.is_running
    assert not backend.is_running
    assert stream.statistics.output_underflow_samples == 0


def test_output_underflow_produces_silence_and_is_observable() -> None:
    backend = FakeDuplexAudioBackend()
    stream = RealtimeAudioStream(backend, sample_rate=1_000, block_size=4)
    stream.write_output(np.array([0.5, 0.25], dtype=np.float32))
    stream.start()

    output = backend.feed(np.ones(4, dtype=np.float32), status="input overflow")

    assert np.array_equal(output, [0.5, 0.25, 0.0, 0.0])
    assert stream.statistics.output_underflow_samples == 2
    assert stream.statistics.callback_statuses == ("input overflow",)


def test_input_and_output_buffers_remain_bounded() -> None:
    backend = FakeDuplexAudioBackend()
    stream = RealtimeAudioStream(
        backend, sample_rate=100, block_size=4, buffer_duration_seconds=0.08
    )
    assert stream.input_buffer.capacity_samples == 8
    stream.start()

    backend.feed(np.arange(12, dtype=np.float32))
    output_dropped = stream.write_output(np.arange(12, dtype=np.float32))

    assert stream.statistics.input_overflow_samples == 4
    assert stream.input_buffer.available_samples == 8
    assert np.array_equal(stream.read_input(8), np.arange(4, 12, dtype=np.float32))
    assert output_dropped == 4
    assert stream.statistics.output_overflow_samples == 4


def test_stream_start_stop_and_fake_backend_lifecycle() -> None:
    backend = FakeDuplexAudioBackend()
    stream = RealtimeAudioStream(backend, sample_rate=16_000, block_size=320)

    with pytest.raises(RuntimeError, match="not running"):
        backend.feed(np.zeros(320, dtype=np.float32))
    stream.start()
    assert stream.is_running
    assert backend.sample_rate == 16_000
    assert backend.block_size == 320
    with pytest.raises(RuntimeError, match="already running"):
        stream.start()
    stream.stop()
    stream.stop()


def test_sounddevice_backend_wires_devices_and_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: dict[str, object] = {}

    class FakePortAudioStream:
        def __init__(self, **kwargs: object) -> None:
            created.update(kwargs)
            created["instance"] = self
            self.started = False
            self.stopped = False
            self.closed = False

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(stream_module.sd, "Stream", FakePortAudioStream)
    backend = SoundDeviceDuplexBackend(input_device=2, output_device="speakers")
    backend.start(16_000, 4, lambda samples, status: samples * 0.5)
    device_callback = created["callback"]
    indata = np.ones((4, 1), dtype=np.float32)
    outdata = np.zeros((4, 1), dtype=np.float32)

    device_callback(indata, outdata, 4, object(), object())
    instance = created["instance"]
    backend.stop()

    assert created["samplerate"] == 16_000
    assert created["blocksize"] == 4
    assert created["device"] == (2, "speakers")
    assert np.array_equal(outdata[:, 0], np.full(4, 0.5, dtype=np.float32))
    assert instance.started
    assert instance.stopped
    assert instance.closed


@pytest.mark.parametrize(
    ("sample_rate", "block_size", "buffer_duration"),
    [(0, 320, 1.0), (16_000, 0, 1.0), (16_000, 320, 0.0)],
)
def test_stream_rejects_invalid_configuration(
    sample_rate: int, block_size: int, buffer_duration: float
) -> None:
    with pytest.raises(ValueError):
        RealtimeAudioStream(
            FakeDuplexAudioBackend(),
            sample_rate=sample_rate,
            block_size=block_size,
            buffer_duration_seconds=buffer_duration,
        )
