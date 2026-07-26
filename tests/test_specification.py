from pathlib import Path


def test_working_spec_starts_as_exact_copy() -> None:
    project_root = Path(__file__).parents[1]
    initial = (project_root / "docs" / "initial_spec.md").read_bytes()
    working = (project_root / "docs" / "spec.md").read_bytes()

    assert initial
    assert working == initial

