"""Fixtures for tests that drive a real LAMMPS build.

The whole directory is skipped when LAMMPS is not importable, so the unit
suite stays green on machines and CI runners without an MD engine.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from nanover_extensions.lammps.simulation import LAMMPSSimulation

pytest.importorskip("lammps")


@pytest.fixture
def two_atoms_input(data_dir: Path) -> Path:
    """Path to the smallest input deck that produces a runnable simulation."""
    return data_dir / "two_atoms.in"


@pytest.fixture
def simulation(two_atoms_input: Path) -> "LAMMPSSimulation":
    from nanover_extensions.lammps.simulation import LAMMPSSimulation

    sim = LAMMPSSimulation(two_atoms_input)
    sim.load()
    return sim
