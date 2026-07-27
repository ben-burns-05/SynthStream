import numpy as np
import pytest

from synthstream.audio import AudioRingBuffer


def test_ring_buffer_preserves_order_across_wraparound() -> None:
    buffer = AudioRingBuffer(5)
    buffer.write(np.array([1, 2, 3], dtype=np.float32))
    assert np.array_equal(buffer.read(2), [1, 2])

    buffer.write(np.array([4, 5, 6, 7], dtype=np.float32))

    assert buffer.available_samples == 5
    assert np.array_equal(buffer.read(5), [3, 4, 5, 6, 7])


def test_ring_buffer_drops_oldest_samples_to_bound_latency() -> None:
    buffer = AudioRingBuffer(4)

    assert buffer.write(np.array([1, 2, 3], dtype=np.float32)) == 0
    assert buffer.write(np.array([4, 5, 6], dtype=np.float32)) == 2
    assert np.array_equal(buffer.read(10), [3, 4, 5, 6])

    assert buffer.write(np.arange(10, dtype=np.float32)) == 6
    assert np.array_equal(buffer.read(4), [6, 7, 8, 9])


def test_padded_read_reports_underflow() -> None:
    buffer = AudioRingBuffer(4)
    buffer.write(np.array([0.25, 0.5], dtype=np.float32))

    samples, missing = buffer.read_padded(4)

    assert missing == 2
    assert np.array_equal(samples, [0.25, 0.5, 0.0, 0.0])


def test_ring_buffer_validates_sizes_and_can_clear() -> None:
    with pytest.raises(ValueError):
        AudioRingBuffer(0)

    buffer = AudioRingBuffer(4)
    buffer.write(np.ones(3, dtype=np.float32))
    buffer.clear()
    assert buffer.available_samples == 0
    with pytest.raises(ValueError):
        buffer.read(-1)

