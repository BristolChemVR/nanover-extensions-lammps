import numpy as np
import pytest

pytest.importorskip("lammps")

from simulation_helpers import make_simulation

from fakes import FakeAppServer, FakeLammps


def test_extract_bonds_ignores_lammps_bond_types() -> None:
    """The LAMMPS bond type is not used in the `Simulation` object, so it should be ignored."""
    lmp = FakeLammps(
        types=[1, 1, 1],
        ids=[10, 20, 30],
        bonds=[[3, 10, 20], [7, 20, 30]],  # bond types 3 and 7 aren't real orders
    )
    sim = make_simulation(lmp=lmp)

    bond_orders, bond_pairs = sim.extract_bonds()

    assert np.array_equal(bond_pairs, np.array([[0, 1], [1, 2]]))
    assert np.array_equal(bond_orders, np.array([1, 1]))  # bond orders are all set to 1


def test_reset_keeps_a_boundary_straddling_bond_for_next_time() -> None:
    """A bond that's hidden this frame because it crosses the box edge must not be lost for good."""
    # 9.8 Å apart in a 10 Å box — real bond, but longer than half the box edge,
    # so _filter_pbc_bonds should hide it from this frame only.
    lmp = FakeLammps(
        types=[1, 1],
        ids=[10, 20],
        positions=[[0.0, 0.0, 0.0], [9.8, 0.0, 0.0]],
        box_bounds=(0.0, 10.0, 0.0, 10.0, 0.0, 10.0),
        bonds=[[1, 10, 20]],
    )
    app_server = FakeAppServer()
    sim = make_simulation(lmp=lmp, generate_bonds=False)

    sim.reset(app_server)

    sent_frame = app_server.frame_publisher.frames[0]
    assert sent_frame.bond_pairs.shape[0] == 0  # hidden this frame, it crosses the boundary

    # ...but the master list has to keep it, or the next reset() loses it for good
    assert np.array_equal(sim._bond_pairs, np.array([[0, 1]]))
    assert np.array_equal(sim._bond_orders, np.array([1]))
