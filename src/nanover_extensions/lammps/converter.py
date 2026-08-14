import numpy as np
from nanover.trajectory import FrameData


def lammps_to_frame_data(
    *,
    positions_nm: np.ndarray | None = None,
    box_bounds_nm: tuple[float, float, float, float, float, float] | None = None,
    particle_count: int | None = None,
    particle_elements: np.ndarray | None = None,
    bond_orders: np.ndarray | None = None,
    bond_pairs: np.ndarray | None = None,
    include_positions: bool = True,
) -> FrameData:
    data = FrameData()

    if include_positions and positions_nm is not None:
        data.particle_positions = np.asarray(positions_nm, dtype=np.float32)

    if box_bounds_nm is not None:
        xlo, xhi, ylo, yhi, zlo, zhi = np.asarray(box_bounds_nm, dtype=float).flatten()[:6]
        data.box_vectors = np.array(
            [[xhi - xlo, 0.0, 0.0], [0.0, yhi - ylo, 0.0], [0.0, 0.0, zhi - zlo]],
            dtype=np.float32,
        )

    if particle_count is not None:
        data.particle_count = int(particle_count)

    if particle_elements is not None:
        elems = np.asarray(particle_elements).reshape((-1,))
        data.particle_elements = elems.astype(np.uint8, copy=False)

    if bond_pairs is not None:
        pairs = np.asarray(bond_pairs).reshape((-1, 2))
        data.bond_pairs = pairs.astype(np.int32, copy=False)

    if bond_orders is not None:
        orders = np.asarray(bond_orders).reshape((-1,))
        data.bond_orders = orders.astype(np.uint32, copy=False)

    return data
