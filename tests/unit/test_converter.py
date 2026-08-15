import numpy as np

from nanover_extensions.lammps.converter import lammps_to_frame_data


def test_box_bounds_become_edge_lengths() -> None:
    frame_data = lammps_to_frame_data(box_bounds_nm=(0.0, 1.0, 0.0, 2.0, 0.0, 3.0))
    np.testing.assert_allclose(np.diag(frame_data.box_vectors), [1.0, 2.0, 3.0])
