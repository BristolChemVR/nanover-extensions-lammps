"""Tests for the frames `LAMMPSSimulation` sends from `advance_to_next_frame`."""

import numpy as np
import pytest

pytest.importorskip("lammps")

from simulation_helpers import make_simulation

from fakes import FakeAppServer, FakeLammps

BOX = (0.0, 10.0, 0.0, 10.0, 0.0, 10.0)


@pytest.fixture
def lmp() -> FakeLammps:
    """Two bonded atoms 9.8 Å apart in a 10 Å box: a real bond that crosses the box edge."""
    return FakeLammps(
        types=[1, 1],
        ids=[10, 20],
        positions=[[0.0, 0.0, 0.0], [9.8, 0.0, 0.0]],
        box_bounds=BOX,
        bonds=[[1, 10, 20]],
    )


def test_a_boundary_straddling_bond_is_hidden_from_the_frame_but_not_forgotten(
    lmp: FakeLammps,
) -> None:
    app_server = FakeAppServer()
    sim = make_simulation(lmp=lmp, generate_bonds=False)
    sim.reset(app_server)

    sim.advance_to_next_frame()

    frame = app_server.frame_publisher.frames[-1]
    assert frame.bond_pairs.shape[0] == 0  # hidden from this frame
    assert np.array_equal(sim._bond_pairs, np.array([[0, 1]]))  # but still in the master list


def test_a_bond_that_stops_straddling_the_boundary_comes_back(lmp: FakeLammps) -> None:
    app_server = FakeAppServer()
    sim = make_simulation(lmp=lmp, generate_bonds=False)
    sim.reset(app_server)
    sim.advance_to_next_frame()

    lmp._positions[1] = [0.8, 0.0, 0.0]  # the atoms have since drifted back together
    sim.advance_to_next_frame()

    frame = app_server.frame_publisher.frames[-1]
    assert np.array_equal(frame.bond_pairs, np.array([[0, 1]]))
