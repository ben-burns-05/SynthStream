import numpy as np

from synthstream.rendering import BufferedOverlapComposer


def test_buffered_overlap_composer_crossfades_and_preserves_nominal_duration() -> None:
    composer = BufferedOverlapComposer(10, staging_seconds=1.0)
    assert len(composer.append(np.ones(4, dtype=np.float32))) == 0

    released = composer.append(np.zeros(4, dtype=np.float32), overlap_samples=2)
    assert len(released) == 0

    result = composer.flush()
    assert len(result) == 6
    assert np.allclose(result, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def test_buffered_overlap_composer_releases_only_immutable_prefix() -> None:
    composer = BufferedOverlapComposer(10, staging_seconds=0.2)
    released = composer.append(np.arange(8, dtype=np.float32))
    assert np.array_equal(released, np.arange(6, dtype=np.float32))
    assert composer.pending_samples == 2
    assert np.array_equal(composer.flush(), np.array([6.0, 7.0], dtype=np.float32))
