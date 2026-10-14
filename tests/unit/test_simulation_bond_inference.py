"""Tests for the bond inference `LAMMPSSimulation.reset` does for atoms with no explicit bonds."""

import numpy as np
import pytest

pytest.importorskip("lammps")

from simulation_helpers import make_simulation

from fakes import FakeAppServer, FakeLammps

BOX = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
HYDROGEN = {1: 1}


def test_inference_only_adds_bonds_touching_atoms_without_explicit_bonds() -> None:
    # Atoms 0 and 1 are bonded explicitly. Atom 2 is not, and sits 0.6 Å from atom 1, so
    # inference finds both (0, 1) and (1, 2): only the one touching atom 2 may be added.
    lmp = FakeLammps(
        types=[1, 1, 1],
        ids=[10, 20, 30],
        positions=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.6], [1.0, 1.0, 2.2]],
        box_bounds=BOX,
        bonds=[[1, 10, 20]],
    )
    sim = make_simulation(lmp=lmp, generate_bonds=True, type_to_atomic_number=HYDROGEN)

    sim.reset(FakeAppServer())

    assert np.array_equal(sim._bond_pairs, np.array([[0, 1], [1, 2]]))
    assert np.array_equal(sim._bond_orders, np.array([1, 1]))


def test_a_system_with_no_explicit_bonds_is_bonded_entirely_by_inference() -> None:
    lmp = FakeLammps(
        types=[1, 1],
        positions=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.6]],
        box_bounds=BOX,
    )
    sim = make_simulation(lmp=lmp, generate_bonds=True, type_to_atomic_number=HYDROGEN)

    sim.reset(FakeAppServer())

    assert np.array_equal(sim._bond_pairs, np.array([[0, 1]]))


def test_generate_bonds_false_leaves_unbonded_atoms_alone() -> None:
    lmp = FakeLammps(
        types=[1, 1],
        positions=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.6]],
        box_bounds=BOX,
    )
    sim = make_simulation(lmp=lmp, generate_bonds=False, type_to_atomic_number=HYDROGEN)

    sim.reset(FakeAppServer())

    assert sim._bond_pairs.shape == (0, 2)
