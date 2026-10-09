"""End-to-end test for hLAMMPS build through `LAMMPSSimulation`."""

import numpy as np

from fakes import FakeAppServer
from nanover_extensions.lammps.simulation import LAMMPSSimulation


def test_reset_publishes_the_topology_of_the_deck(simulation: LAMMPSSimulation) -> None:
    app_server = FakeAppServer()

    simulation.reset(app_server)

    frame = app_server.frame_publisher.frames[0]
    assert frame.particle_count == 2
    assert np.array_equal(frame.particle_elements, [1, 1])  # 1.008 amu is hydrogen
    assert np.array_equal(frame.bond_pairs, [[0, 1]])
    np.testing.assert_allclose(
        frame.particle_positions, [[0.45, 0.5, 0.5], [0.55, 0.5, 0.5]], atol=1e-5
    )  # 4.5 Å and 5.5 Å in a 10 Å box, in nm
