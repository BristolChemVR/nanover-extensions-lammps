"""Fixtures shared by the whole test suite."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Directory holding the small checked-in LAMMPS input decks."""
    return Path(__file__).parent / "data"
