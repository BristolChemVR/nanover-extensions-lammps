import ctypes
import warnings
from pathlib import Path
from typing import Any

import lammps
import numpy as np

from nanover_extensions.lammps.converter import lammps_to_frame_data
from nanover_extensions.lammps.imd import (LammpsImdForceManager,
                                           detect_lammps_units,
                                           get_unit_conversions)

# radii from Alvarez, Dalton Trans. 2008, 2832 (single-bond covalent radii)
_RADII_BY_Z: dict[int, float] = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    11: 1.66,
    12: 1.41,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    19: 2.03,
    20: 1.76,
    26: 1.52,
    29: 1.32,
    30: 1.22,
    35: 1.20,
    53: 1.39,
}
_DEFAULT_RADIUS = 0.80  # Å fallback for unlisted elements
_BOND_FACTOR = 1.15


class LAMMPSSimulation:
    """LAMMPS simulation wrapper implementing the Simulation protocol."""

    def __init__(
        self,
        input_script: str | Path,
        type_to_atomic_number: dict[int, int] | None = None,
        lammps_units: str | None = None,
        frame_interval_steps: int = 1,
        *,
        include_velocities: bool = False,
        include_forces: bool = False,
        generate_bonds: bool = True,
        quiet: bool = False,
    ) -> None:
        self.input_script = input_script
        self.include_velocities = include_velocities
        self.include_forces = include_forces
        self.generate_bonds = generate_bonds
        self.frame_interval = frame_interval_steps
        self.type_to_atomic_number = type_to_atomic_number or {}
        self.name = Path(input_script).stem

        cmdargs = ["-screen", "none"] if quiet else []
        self.lmp = lammps.lammps(cmdargs=cmdargs)
        self.lmp.file(self.input_script)

        # Detect or accept LAMMPS unit style (needed for IMD force conversion)
        self.lammps_units: str = lammps_units or detect_lammps_units(self.lmp)

        # Cache periodicity flags and position unit conversion factor once after
        # the script is loaded — neither changes during a run.
        xp = int(self.lmp.extract_global("xperiodic") or 0)
        yp = int(self.lmp.extract_global("yperiodic") or 0)
        zp = int(self.lmp.extract_global("zperiodic") or 0)
        self._is_periodic = np.array([xp, yp, zp], dtype=bool)
        self._pos_to_nm, _, _ = get_unit_conversions(self.lammps_units)

        self._app_server = None
        self._current_step = 0

        self._id_to_index: dict[int, int] | None = None
        self._bond_pairs: np.ndarray | None = None
        self._bond_orders: np.ndarray | None = None
        self._particle_elements: np.ndarray | None = None

        self._imd_force_manager: LammpsImdForceManager | None = None
        self._needs_pre: bool = True  # True after reset() until first step()

    def step(self, n: int = 1) -> None:
        if self._needs_pre:
            self.lmp.command(f"run {int(n)} post no")
            self._needs_pre = False
        else:
            self.lmp.command(f"run {int(n)} pre no post no")

    def load(self) -> None:
        pass

    def _build_id_to_index_map(self) -> dict[int, int]:
        ids = np.asarray(self.lmp.gather_atoms("id", 0, 1), dtype=np.int64)
        return {int(aid): i for i, aid in enumerate(ids)}

    @staticmethod
    def _filter_pbc_bonds(
        positions: np.ndarray,
        box_bounds: tuple[float, float, float, float, float, float],
        bond_pairs: np.ndarray | None,
        bond_orders: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Drop bonds longer than half the shortest box edge — these are PBC artefacts."""
        if bond_pairs is None or bond_orders is None or len(bond_pairs) == 0:
            return bond_pairs, bond_orders
        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds
        min_half_length = min(xhi - xlo, yhi - ylo, zhi - zlo) * 0.5
        delta = positions[bond_pairs[:, 0]] - positions[bond_pairs[:, 1]]
        keep = np.linalg.norm(delta, axis=1) < min_half_length
        return bond_pairs[keep], bond_orders[keep]

    def reset(self, app_server: Any | None = None) -> None:
        if app_server is not None:
            self._app_server = app_server

        if self._imd_force_manager is not None:
            self._imd_force_manager.unfix()
            self._imd_force_manager = None

        if self._id_to_index is None:
            self._id_to_index = self._build_id_to_index_map()

        if self._particle_elements is None:
            self._particle_elements = self._build_particle_elements()

        if self._bond_pairs is None or self._bond_orders is None:
            bond_orders, bond_pairs = self.extract_bonds()
            if self.generate_bonds:
                natoms = int(self.lmp.get_natoms())
                # Find atoms that have no explicit LAMMPS bond (e.g. zeolite Si/O
                # framework in a simulation that also has an organic guest molecule
                # with explicit bonds).
                bonded_atoms: set[int] = (
                    set(bond_pairs.ravel().tolist()) if len(bond_pairs) > 0 else set()
                )
                if len(bonded_atoms) < natoms:
                    raw_pos = np.asarray(
                        self.lmp.gather_atoms("x", 1, 3),
                        dtype=float,
                    ).reshape((natoms, 3))
                    # Convert positions to Å regardless of LAMMPS unit style
                    # so that the bond-distance thresholds (which are in Å) are correct.
                    pos_to_nm, _, _ = get_unit_conversions(self.lammps_units)
                    to_angstrom = pos_to_nm * 10.0
                    # wrap into the primary cell first — gather_atoms("x") isn't
                    # guaranteed to be in [xlo, xhi), which would break bond inference
                    _box = self.lmp.extract_box()
                    _lo = np.array(
                        [float(_box[0][0]), float(_box[0][1]), float(_box[0][2])],
                        dtype=float,
                    )
                    _box_lengths = np.array(
                        [
                            float(_box[1][0]) - float(_box[0][0]),
                            float(_box[1][1]) - float(_box[0][1]),
                            float(_box[1][2]) - float(_box[0][2]),
                        ],
                        dtype=float,
                    )
                    raw_pos = (raw_pos - _lo) % _box_lengths + _lo
                    # box_lengths=None: a PBC-straddling bond would render as a line
                    # across the whole cell, so we'd rather miss a few edge bonds
                    extra_orders, extra_pairs = self._generate_bonds_from_positions(
                        raw_pos * to_angstrom,
                        self._particle_elements,
                        box_lengths=None,
                    )
                    if len(extra_pairs) > 0:
                        # only keep inferred bonds touching a previously-unbonded atom
                        unbonded = np.array(
                            sorted(set(range(natoms)) - bonded_atoms),
                            dtype=np.int32,
                        )
                        mask = np.isin(extra_pairs[:, 0], unbonded) | np.isin(
                            extra_pairs[:, 1],
                            unbonded,
                        )
                        extra_pairs = extra_pairs[mask]
                        extra_orders = extra_orders[mask]
                    if len(extra_pairs) > 0:
                        bond_pairs = (
                            np.vstack([bond_pairs, extra_pairs])
                            if len(bond_pairs) > 0
                            else extra_pairs
                        )
                        bond_orders = (
                            np.concatenate([bond_orders, extra_orders])
                            if len(bond_orders) > 0
                            else extra_orders
                        )
            self._bond_pairs = bond_pairs
            self._bond_orders = bond_orders

        positions, box_bounds = self._get_positions_and_box()
        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds

        self._bond_pairs, self._bond_orders = self._filter_pbc_bonds(
            positions,
            box_bounds,
            self._bond_pairs,
            self._bond_orders,
        )

        if self._is_periodic.all():
            pbc_vectors = np.diag(
                [
                    (xhi - xlo) * self._pos_to_nm,
                    (yhi - ylo) * self._pos_to_nm,
                    (zhi - zlo) * self._pos_to_nm,
                ],
            )
        else:
            pbc_vectors = None

        imd_state = self._app_server.imd if self._app_server is not None else None
        self._imd_force_manager = LammpsImdForceManager(
            lmp=self.lmp,
            imd_state=imd_state,
            id_to_index=self._id_to_index,
            pbc_vectors=pbc_vectors,
            lammps_units=self.lammps_units,
        )
        # New fix registered — next run must use pre yes to incorporate it.
        self._needs_pre = True

        if self._app_server is not None:
            natoms = int(self.lmp.get_natoms())
            xlo, xhi, ylo, yhi, zlo, zhi = box_bounds
            topology_frame = lammps_to_frame_data(
                positions_nm=positions * self._pos_to_nm,
                box_bounds_nm=(
                    xlo * self._pos_to_nm,
                    xhi * self._pos_to_nm,
                    ylo * self._pos_to_nm,
                    yhi * self._pos_to_nm,
                    zlo * self._pos_to_nm,
                    zhi * self._pos_to_nm,
                ),
                particle_count=natoms,
                particle_elements=self._particle_elements,
                bond_pairs=self._bond_pairs,
                bond_orders=self._bond_orders,
                include_positions=True,
            )
            self._app_server.frame_publisher.send_clear()
            self._app_server.frame_publisher.send_frame(topology_frame)

    def advance_by_one_step(self) -> None:
        self.advance_to_next_frame()

    def advance_by_seconds(self, dt: float) -> None:
        self.advance_to_next_frame()

    def _get_positions_and_box(
        self,
    ) -> tuple[np.ndarray, tuple[float, float, float, float, float, float]]:
        natoms = int(self.lmp.get_natoms())

        box = self.lmp.extract_box()
        xlo, ylo, zlo = float(box[0][0]), float(box[0][1]), float(box[0][2])
        xhi, yhi, zhi = float(box[1][0]), float(box[1][1]), float(box[1][2])
        box_bounds = (xlo, xhi, ylo, yhi, zlo, zhi)

        positions = np.asarray(
            self.lmp.gather_atoms("x", 1, 3),
            dtype=float,
        ).reshape((natoms, 3))

        _box_lengths = np.array([xhi - xlo, yhi - ylo, zhi - zlo], dtype=float)
        origin = np.array([xlo, ylo, zlo], dtype=float)
        shifted = positions - origin
        if self._is_periodic.any():
            positions = np.where(self._is_periodic, shifted % _box_lengths, shifted)
        else:
            positions = shifted

        return positions, box_bounds

    def _build_particle_elements(self) -> np.ndarray:
        """Map LAMMPS per-atom type to atomic number (uint8), via explicit overrides or mass."""
        natoms = int(self.lmp.get_natoms())
        lmp_types = np.asarray(
            self.lmp.gather_atoms("type", 0, 1), dtype=np.int32
        ).reshape(
            (natoms,),
        )

        # (atomic_number, atomic_weight) reference table, extend as needed
        mass_table: list[tuple[int, float]] = [
            (1, 1.00794),  # H
            (6, 12.0107),  # C
            (7, 14.0067),  # N
            (8, 15.9994),  # O
            (9, 18.9984),  # F
            (11, 22.9898),  # Na
            (12, 24.3050),  # Mg
            (13, 26.9815),  # Al
            (14, 28.0855),  # Si
            (15, 30.9738),  # P
            (16, 32.065),  # S
            (17, 35.453),  # Cl
            (19, 39.0983),  # K
            (20, 40.078),  # Ca
            (26, 55.845),  # Fe
            (29, 63.546),  # Cu
            (30, 65.38),  # Zn
            (35, 79.904),  # Br
            (53, 126.904),  # I
        ]

        def closest_z_from_mass(m: float, tol: float = 0.6) -> int | None:
            """Return closest atomic number by mass if within tolerance, else None."""
            best_z: int | None = None
            best_diff = float("inf")
            for z, ref_m in mass_table:
                d = abs(m - ref_m)
                if d < best_diff:
                    best_diff = d
                    best_z = z
            if best_z is None or best_diff > tol:
                return None
            return best_z

        # Infer number of types from observed types (safe across LAMMPS builds)
        ntypes = int(lmp_types.max(initial=0))

        # LAMMPS per-type arrays are 1-based (mass[1..ntypes])
        masses = None
        try:
            mass_ptr = self.lmp.extract_atom("mass", 2)  # pointer to double array
            if mass_ptr is not None and ntypes > 0:
                masses = np.ctypeslib.as_array(
                    ctypes.cast(mass_ptr, ctypes.POINTER(ctypes.c_double)),
                    shape=(ntypes + 1,),
                )
        except Exception as e:
            warnings.warn(
                f"Could not read per-type masses; elements inferred from mass will be missing: {e}",
                stacklevel=2,
            )
            masses = None

        # Fill in any missing type->Z mapping using masses.
        if masses is not None:
            for atom_type in range(1, ntypes + 1):
                if int(atom_type) in self.type_to_atomic_number:
                    continue  # explicit override
                m = float(masses[atom_type])
                z = closest_z_from_mass(m)
                if z is not None:
                    self.type_to_atomic_number[int(atom_type)] = int(z)

        out = np.empty(natoms, dtype=np.uint8)
        for i, t in enumerate(lmp_types):
            z = self.type_to_atomic_number.get(int(t))
            out[i] = np.uint8(0 if z is None else max(0, min(255, int(z))))
        return out

    def extract_bonds(self) -> tuple[np.ndarray, np.ndarray]:
        bonds = self.lmp.numpy.gather_bonds()  # [type, id1, id2]

        if self._id_to_index is None:
            self._id_to_index = self._build_id_to_index_map()

        bond_types = bonds[:, 0].astype(np.int32)
        id1 = bonds[:, 1]
        id2 = bonds[:, 2]

        idx1 = np.array([self._id_to_index[int(x)] for x in id1], dtype=np.int32)
        idx2 = np.array([self._id_to_index[int(x)] for x in id2], dtype=np.int32)

        i = np.minimum(idx1, idx2)
        j = np.maximum(idx1, idx2)
        pairs = np.stack([i, j], axis=1)

        return bond_types, pairs

    @staticmethod
    def _generate_bonds_from_positions(
        positions: np.ndarray,
        elements: np.ndarray,
        box_lengths: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Infer bonds when distance < 1.15 * (covalent radius sum). Positions in Å."""
        n = len(positions)
        radii = np.array([_RADII_BY_Z.get(int(z), _DEFAULT_RADIUS) for z in elements])

        bond_pairs: list[tuple[int, int]] = []
        for i in range(n):
            diffs = positions[i + 1 :] - positions[i]  # (n-i-1, 3)
            if box_lengths is not None:
                diffs -= np.round(diffs / box_lengths) * box_lengths
            dists = np.linalg.norm(diffs, axis=1)
            cutoffs = _BOND_FACTOR * (radii[i] + radii[i + 1 :])
            bond_pairs.extend((i, i + 1 + k) for k in np.where(dists < cutoffs)[0])

        if not bond_pairs:
            return np.empty(0, dtype=np.int32), np.empty((0, 2), dtype=np.int32)

        pairs = np.array(bond_pairs, dtype=np.int32)
        orders = np.ones(len(pairs), dtype=np.int32)
        return orders, pairs

    def advance_to_next_frame(self) -> None:
        """Advance the simulation by frame_interval steps and send a frame to the app server."""
        self.step(self.frame_interval)
        self._current_step += self.frame_interval

        positions, box_bounds = self._get_positions_and_box()

        if self._imd_force_manager is not None:
            self._imd_force_manager.update_interactions(
                positions_nm=positions * self._pos_to_nm
            )

        vis_pairs, vis_orders = self._filter_pbc_bonds(
            positions,
            box_bounds,
            self._bond_pairs,
            self._bond_orders,
        )

        xlo, xhi, ylo, yhi, zlo, zhi = box_bounds
        frame = lammps_to_frame_data(
            positions_nm=positions * self._pos_to_nm,
            box_bounds_nm=(
                xlo * self._pos_to_nm,
                xhi * self._pos_to_nm,
                ylo * self._pos_to_nm,
                yhi * self._pos_to_nm,
                zlo * self._pos_to_nm,
                zhi * self._pos_to_nm,
            ),
            bond_pairs=vis_pairs,
            bond_orders=vis_orders,
            include_positions=True,
        )

        if self._imd_force_manager is not None:
            self._imd_force_manager.add_to_frame_data(frame)

        if self._app_server is not None:
            self._app_server.frame_publisher.send_frame(frame)
