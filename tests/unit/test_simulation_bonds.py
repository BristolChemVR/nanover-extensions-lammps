import numpy as np
import pytest

# `simulation.py` imports `lammps` at module scope, so this file cannot run
# without a LAMMPS build. See the note in the README about moving
# `_generate_bonds_from_positions` into a LAMMPS-free module.
pytest.importorskip("lammps")

from nanover_extensions.lammps.simulation import LAMMPSSimulation

# Two hydrogens bond below 1.15 * (0.31 + 0.31) = 0.713 Å.
HYDROGEN = np.array([1, 1])


@pytest.fixture
def two_hydrogens_bonded() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.6]])


@pytest.fixture
def two_hydrogens_apart() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]])


def test_close_hydrogen_atoms_are_bonded(two_hydrogens_bonded: np.ndarray) -> None:
    orders, pairs = LAMMPSSimulation._generate_bonds_from_positions(
        two_hydrogens_bonded, HYDROGEN, None
    )
    np.testing.assert_array_equal(pairs, [[0, 1]])
    np.testing.assert_array_equal(orders, [1])


def test_distant_hydrogen_atoms_are_not_bonded(two_hydrogens_apart: np.ndarray) -> None:
    orders, pairs = LAMMPSSimulation._generate_bonds_from_positions(
        two_hydrogens_apart, HYDROGEN, None
    )
    assert pairs.shape == (0, 2)
    assert orders.shape == (0,)


def test_bonds_are_found_across_periodic_boundary() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 9.8]])
    box_lengths = np.array([10.0, 10.0, 10.0])
    _orders, pairs = LAMMPSSimulation._generate_bonds_from_positions(
        positions, HYDROGEN, box_lengths
    )
    np.testing.assert_array_equal(pairs, [[0, 1]])


def test_same_atoms_are_not_bonded_without_periodic_boundary() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 9.8]])
    _orders, pairs = LAMMPSSimulation._generate_bonds_from_positions(positions, HYDROGEN, None)
    assert pairs.shape == (0, 2)
