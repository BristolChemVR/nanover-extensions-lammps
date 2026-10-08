"""Tests for how `LAMMPSSimulation` decides which element each LAMMPS atom type is."""

import numpy as np
import pytest

pytest.importorskip("lammps")

from simulation_helpers import make_simulation

from fakes import FakeLammps

# Two hydrogens and an oxygen
TYPES = [1, 1, 2]
MASSES_BY_TYPE = [0.0, 1.008, 15.999]  # index 0 is LAMMPS' unused padding slot


def test_elements_are_inferred_from_per_type_masses() -> None:
    lmp = FakeLammps(types=TYPES, masses_by_type=MASSES_BY_TYPE)
    sim = make_simulation(lmp=lmp)

    elements = sim._build_particle_elements()

    assert elements.dtype == np.uint8
    np.testing.assert_array_equal(elements, [1, 1, 8])


def test_a_mass_within_tolerance_still_matches_its_element() -> None:
    # 1.5 is 0.49 away from hydrogen (1.008), inside the 0.6 tolerance
    lmp = FakeLammps(types=[1], masses_by_type=[0.0, 1.5])
    sim = make_simulation(lmp=lmp)

    np.testing.assert_array_equal(sim._build_particle_elements(), [1])


def test_a_mass_outside_tolerance_gives_element_zero() -> None:
    # 3.0 is ~2 away from hydrogen and nothing else is closer, so it must not be guessed
    lmp = FakeLammps(types=[1], masses_by_type=[0.0, 3.0])
    sim = make_simulation(lmp=lmp)

    np.testing.assert_array_equal(sim._build_particle_elements(), [0])


def test_an_explicit_override_beats_the_mass() -> None:
    # Type 2 weighs the same as oxygen, but the user says it is sulfur
    lmp = FakeLammps(types=TYPES, masses_by_type=MASSES_BY_TYPE)
    sim = make_simulation(lmp=lmp, type_to_atomic_number={2: 16})

    np.testing.assert_array_equal(sim._build_particle_elements(), [1, 1, 16])


def test_an_override_is_used_when_there_are_no_masses_to_read() -> None:
    lmp = FakeLammps(types=TYPES)  # no per-type mass table at all
    sim = make_simulation(lmp=lmp, type_to_atomic_number={1: 14, 2: 8})

    np.testing.assert_array_equal(sim._build_particle_elements(), [14, 14, 8])


def test_without_masses_or_overrides_every_element_is_zero() -> None:
    lmp = FakeLammps(types=TYPES)
    sim = make_simulation(lmp=lmp)

    np.testing.assert_array_equal(sim._build_particle_elements(), [0, 0, 0])


def test_unreadable_masses_warn_and_leave_elements_unset() -> None:
    lmp = FakeLammps(types=TYPES, masses_by_type=MASSES_BY_TYPE, unreadable=["mass"])
    sim = make_simulation(lmp=lmp)

    with pytest.warns(UserWarning, match="Could not read per-type masses"):
        elements = sim._build_particle_elements()

    np.testing.assert_array_equal(elements, [0, 0, 0])


def test_unreadable_masses_do_not_discard_explicit_overrides() -> None:
    lmp = FakeLammps(types=TYPES, masses_by_type=MASSES_BY_TYPE, unreadable=["mass"])
    sim = make_simulation(lmp=lmp, type_to_atomic_number={1: 1})

    with pytest.warns(UserWarning, match="Could not read per-type masses"):
        elements = sim._build_particle_elements()

    np.testing.assert_array_equal(elements, [1, 1, 0])


def test_elements_follow_atom_order_not_type_order() -> None:
    lmp = FakeLammps(types=[2, 1, 2, 1], masses_by_type=MASSES_BY_TYPE)
    sim = make_simulation(lmp=lmp)

    np.testing.assert_array_equal(sim._build_particle_elements(), [8, 1, 8, 1])
