import numpy as np
import pytest
from nanover.trajectory import MissingDataError

from nanover_extensions.lammps.converter import lammps_to_frame_data


def test_box_bounds_become_edge_lengths() -> None:
    frame_data = lammps_to_frame_data(box_bounds_nm=(0.0, 1.0, 0.0, 2.0, 0.0, 3.0))
    np.testing.assert_allclose(np.diag(frame_data.box_vectors), [1.0, 2.0, 3.0])


def test_positions_are_narrowed_to_float32() -> None:
    frame_data = lammps_to_frame_data(positions_nm=np.array([[0.0, 1.0, 2.0]], dtype=np.float64))
    assert frame_data.particle_positions.dtype == np.float32
    np.testing.assert_allclose(frame_data.particle_positions, [[0.0, 1.0, 2.0]])


def test_positions_are_accepted_as_plain_sequence() -> None:
    frame_data = lammps_to_frame_data(positions_nm=[[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])  # type: ignore[arg-type]
    assert frame_data.particle_positions.dtype == np.float32
    np.testing.assert_allclose(
        frame_data.particle_positions, [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]], atol=1e-6
    )


def test_positions_are_omitted_when_not_included() -> None:
    frame_data = lammps_to_frame_data(
        positions_nm=np.zeros((2, 3)),
        particle_count=2,
        include_positions=False,
    )
    assert frame_data.particle_count == 2
    with pytest.raises(MissingDataError):
        _ = frame_data.particle_positions


def test_flat_bond_pairs_are_rehshaped_into_index_pairs() -> None:
    frame_data = lammps_to_frame_data(bond_pairs=np.array([0, 1, 2, 1]))
    np.testing.assert_array_equal(frame_data.bond_pairs, [[0, 1], [2, 1]])


def test_bond_pairs_keep_their_atom_order() -> None:
    frame_data = lammps_to_frame_data(bond_pairs=np.array([1, 0, 1, 1]))
    np.testing.assert_array_equal(frame_data.bond_pairs, [[1, 0], [1, 1]])


def test_particle_elements_are_stored_as_uint8() -> None:
    frame_data = lammps_to_frame_data(particle_elements=np.array([1, 2, 3]))
    assert frame_data.particle_elements.dtype == np.uint8
    np.testing.assert_array_equal(frame_data.particle_elements, [1, 2, 3])


def test_particle_elements_are_flattened() -> None:
    frame_data = lammps_to_frame_data(particle_elements=np.array([[1], [2], [3]]))
    assert frame_data.particle_elements.shape == (3,)
    np.testing.assert_array_equal(frame_data.particle_elements, [1, 2, 3])


def test_atomic_number_beyond_uint8_wraps_not_clamps() -> None:
    frame_data = lammps_to_frame_data(particle_elements=np.array([300]))
    assert frame_data.particle_elements[0] == 300 % 256


def test_box_vectors_have_no_off_diagonal_components() -> None:
    frame_data = lammps_to_frame_data(box_bounds_nm=(0.0, 1.0, 0.0, 2.0, 0.0, 3.0))
    off_diagonal = frame_data.box_vectors - np.diag(np.diagonal(frame_data.box_vectors))
    np.testing.assert_allclose(off_diagonal, 0.0)


def test_box_edge_lengths_ignore_the_origin_offset() -> None:
    frame_data = lammps_to_frame_data(box_bounds_nm=(-1.0, 0.0, -2.0, 0.0, -3.0, 0.0))
    np.testing.assert_allclose(np.diag(frame_data.box_vectors), [1.0, 2.0, 3.0])


def test_bond_orders_are_flattened_and_stored_as_uint32() -> None:
    frame_data = lammps_to_frame_data(bond_orders=np.array([[1], [2], [3]], dtype=np.uint32))
    assert frame_data.bond_orders.dtype == np.uint32
    np.testing.assert_array_equal(frame_data.bond_orders, [1, 2, 3])


def test_bond_pairs_are_stored_as_int32() -> None:
    frame_data = lammps_to_frame_data(bond_pairs=np.array([0, 1, 2, 3], dtype=np.int32))
    assert frame_data.bond_pairs.dtype == np.int32
    np.testing.assert_array_equal(frame_data.bond_pairs, [[0, 1], [2, 3]])


def test_particle_count_is_stored_as_a_python_int() -> None:
    frame_data = lammps_to_frame_data(particle_count=42)
    assert isinstance(frame_data.particle_count, int)
    assert frame_data.particle_count == 42


def test_box_bounds_are_flattened_before_unpacking() -> None:
    frame_data = lammps_to_frame_data(box_bounds_nm=[[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])  # type: ignore[arg-type]
    np.testing.assert_allclose(np.diag(frame_data.box_vectors), [1.0, 2.0, 3.0])
