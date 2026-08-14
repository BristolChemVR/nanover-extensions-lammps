"""Manage NanoVer IMD force injection into a LAMMPS simulation via fix external."""

import contextlib
import ctypes
import warnings
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from nanover.imd import ImdStateWrapper
from nanover.imd.imd_force import calculate_imd_force, get_sparse_forces
from nanover.trajectory import FrameData

if TYPE_CHECKING:
    import lammps

# Conversion factors per LAMMPS unit style.
# Each entry is (pos_to_nm, force_from_kjmol_per_nm, mass_to_amu):
#   pos_to_nm               : multiply LAMMPS positions by this to get nm
#   force_from_kjmol_per_nm : multiply kJ/(mol·nm) forces by this to get LAMMPS force units
#   mass_to_amu             : multiply LAMMPS masses by this to get atomic mass units
_UNIT_CONVERSIONS: dict[str, tuple[float, float, float]] = {
    # real  : positions Å, forces kcal/(mol·Å), masses g/mol (= amu)
    #   1 kJ/(mol·nm) = 0.1 kJ/(mol·Å) = 0.023901 kcal/(mol·Å)
    "real": (0.1, 0.023901, 1.0),
    # metal : positions Å, forces eV/Å, masses g/mol (= amu)
    #   1 kJ/(mol·nm) = 1/(96.485 * 10) eV/Å ≈ 1.03643e-3 eV/Å
    "metal": (0.1, 1.03643e-3, 1.0),
    # si    : positions m, forces N (per atom), masses kg
    #   1 kJ/(mol·nm) = 1e3/(6.02214076e23 * 1e-9) N ≈ 1.66054e-12 N
    #   1 kg = 1/(1.66054e-27) amu ≈ 6.02214076e26 amu
    "si": (1e9, 1.66054e-12, 6.02214076e26),
    # nano  : positions nm (already), forces ag·nm/ns², masses ag
    #   1 kJ/(mol·nm) ≈ 0.069477 ag·nm/ns²
    #   1 ag = 1e-18 g / (1.66054e-24 g/amu) ≈ 6.02214076e5 amu
    "nano": (1.0, 0.069477, 6.02214076e5),
}


def get_unit_conversions(lammps_units: str) -> tuple[float, float, float]:
    """Return (pos_to_nm, force_from_kjmol_per_nm, mass_to_amu) for a LAMMPS unit style."""
    try:
        return _UNIT_CONVERSIONS[lammps_units]
    except KeyError:
        raise ValueError(
            f"Unsupported LAMMPS unit style '{lammps_units}'. "
            f"Supported styles: {list(_UNIT_CONVERSIONS)}",
        )


def detect_lammps_units(lmp: "lammps.lammps") -> str:
    """Detect the LAMMPS unit style, falling back to "real" if detection fails."""
    try:
        units = lmp.extract_global("units")
        if isinstance(units, (bytes, bytearray)):
            units = units.decode()
    except Exception as e:
        warnings.warn(
            f"Could not detect LAMMPS unit style, assuming 'real': {e}", stacklevel=2
        )
        return "real"

    if isinstance(units, str) and units in _UNIT_CONVERSIONS:
        return units

    warnings.warn(
        f"Unsupported or undetected LAMMPS unit style {units!r}, assuming 'real'.",
        stacklevel=2,
    )
    return "real"


class LammpsImdForceManager:
    FIX_ID = "imd_nanover"

    def __init__(
        self,
        lmp: "lammps.lammps",
        imd_state: ImdStateWrapper | None,
        id_to_index: dict[int, int],
        pbc_vectors: np.ndarray | None = None,
        lammps_units: str | None = None,
    ) -> None:
        self.lmp = lmp
        self.imd_state = imd_state
        self._id_to_index = id_to_index  # LAMMPS atom ID → NanoVer 0-based index

        # _id_lookup[lammps_id] = nanover_index, or -1 if not present.
        max_id = max(id_to_index.keys(), default=0)
        self._id_lookup = np.full(max_id + 1, -1, dtype=np.int64)
        for lammps_id, nanover_idx in id_to_index.items():
            self._id_lookup[lammps_id] = nanover_idx

        if lammps_units is None:
            lammps_units = detect_lammps_units(lmp)
        _, self._force_from_kjmol_nm, self._mass_to_amu = get_unit_conversions(
            lammps_units
        )

        self._masses: np.ndarray = self._get_masses(len(id_to_index))

        # Forces applied by the callback each timestep, LAMMPS units, NanoVer index order.
        self._current_lammps_forces: np.ndarray | None = None

        # Forces in NanoVer units (kJ/mol/nm) for frame broadcasting.
        self.user_forces: np.ndarray = np.empty(0)
        self.total_user_energy: float = 0.0
        self._is_force_dirty: bool = False

        self.periodic_box_lengths: np.ndarray | None = None
        if pbc_vectors is not None:
            pbc = np.asarray(pbc_vectors)
            if not np.all(pbc == np.diagflat(np.diag(pbc))):
                warnings.warn(
                    "The periodic box vectors do not correspond to an orthorhombic cell. "
                    "Only orthorhombic PBC is currently supported for LAMMPS IMD.",
                    stacklevel=2,
                )

            self.periodic_box_lengths = np.diag(pbc)

        # Register fix external — callback is invoked every timestep inside run
        lmp.command(f"fix {self.FIX_ID} all external pf/callback 1 1")
        lmp.set_fix_external_callback(self.FIX_ID, self._callback)

    def update_interactions(self, positions_nm: np.ndarray) -> None:
        """Compute IMD forces from this frame's positions and cache them for _callback.

        Call once per frame using the same positions broadcast to clients, so
        dragging gestures pull in the direction the client actually sees.
        """
        if self.imd_state is None:
            return

        natoms = len(positions_nm)
        interactions = self.imd_state.active_interactions

        if not interactions:
            if self._is_force_dirty:
                self._current_lammps_forces = None
                # _is_force_dirty stays True so _callback zeros fexternal once more
                self.user_forces = np.zeros((natoms, 3), dtype=np.float32)
                self.total_user_energy = 0.0
            return

        energy, forces_kjmol = calculate_imd_force(
            positions_nm,
            self._masses,
            interactions.values(),
            self.periodic_box_lengths,
        )

        # Cache forces converted to LAMMPS units for the callback to apply
        self._current_lammps_forces = (
            np.asarray(forces_kjmol) * self._force_from_kjmol_nm
        )

        self._is_force_dirty = True
        self.total_user_energy = float(energy)
        self.user_forces = np.asarray(forces_kjmol, dtype=np.float32)

    def add_to_frame_data(self, frame_data: FrameData) -> None:
        """Write IMD user energy and forces into *frame_data* for broadcasting."""
        frame_data.user_energy = self.total_user_energy
        if self.user_forces.size > 0:
            sparse_indices, sparse_forces = get_sparse_forces(self.user_forces)
            frame_data.user_forces_sparse = sparse_forces
            frame_data.user_forces_index = sparse_indices

    def unfix(self) -> None:
        """Remove ``fix imd_nanover`` from LAMMPS.  Call before re-initialising."""
        with contextlib.suppress(Exception):
            self.lmp.command(f"unfix {self.FIX_ID}")

    def _callback(
        self,
        lmp: "lammps.lammps",
        ntimestep: int,
        nlocal: int,
        tag: list,
        x: list,
        fexternal: np.ndarray,
    ) -> None:
        """LAMMPS fix-external callback: scatter cached forces into fexternal each timestep."""
        if self._current_lammps_forces is None:
            # zero fexternal even with no interactions — atom migration can regrow it uninitialised
            fexternal[:] = 0.0
            self._is_force_dirty = False
            return

        # Map local atom IDs to NanoVer indices and scatter cached forces
        tag_arr = np.asarray(tag)
        nanover_indices = self._id_lookup[tag_arr]
        valid = nanover_indices >= 0

        fexternal[:] = 0.0
        fexternal[valid] = self._current_lammps_forces[nanover_indices[valid]]

    def _get_masses(self, natoms: int) -> np.ndarray:
        """Return per-atom masses in a.m.u., NanoVer atom order."""
        try:
            lmp_types = np.asarray(self.lmp.gather_atoms("type", 0, 1), dtype=np.int32)
            ntypes = int(lmp_types.max(initial=0))
            mass_ptr = self.lmp.extract_atom("mass", 2)
            if mass_ptr is not None and ntypes > 0:
                masses_by_type = np.ctypeslib.as_array(
                    ctypes.cast(mass_ptr, ctypes.POINTER(ctypes.c_double)),
                    shape=(ntypes + 1,),
                )
                return (
                    np.array([float(masses_by_type[int(t)]) for t in lmp_types])
                    * self._mass_to_amu
                )
        except Exception as e:
            warnings.warn(
                f"Could not read per-type masses, trying per-atom rmass: {e}",
                stacklevel=2,
            )

        # per-atom mass, e.g. sphere/granular atom styles use rmass instead of per-type mass
        try:
            rmass_ptr = self.lmp.extract_atom("rmass", 2)
            if rmass_ptr is not None:
                rmass = np.ctypeslib.as_array(
                    ctypes.cast(rmass_ptr, ctypes.POINTER(ctypes.c_double)),
                    shape=(natoms,),
                )
                return rmass.copy() * self._mass_to_amu
        except Exception as e:
            warnings.warn(f"Could not read per-atom rmass: {e}", stacklevel=2)

        warnings.warn(
            "Could not determine atom masses; using unit masses. "
            "Mass-weighted IMD interactions will be inaccurate.",
            stacklevel=2,
        )
        return np.ones(natoms, dtype=np.float64)
