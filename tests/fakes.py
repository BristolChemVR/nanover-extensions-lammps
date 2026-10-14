"""Test doubles standing in for the real LAMMPS Python bindings."""

import ctypes
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np


def _as_double_array(values: Sequence[float] | None) -> "ctypes.Array[ctypes.c_double] | None":
    """Convert python list into ctypes array that lammps can read."""
    if values is None:
        return None
    return (ctypes.c_double * len(values))(*(float(v) for v in values))


class _FakeLammpsNumpy:
    """Stand-in for the `lmp.numpy` accessor, which only `gather_bonds` needs so far."""

    def __init__(self, bonds: Sequence[Sequence[int]] | None) -> None:
        self._bonds = np.asarray(bonds if bonds is not None else [], dtype=np.int64).reshape(-1, 3)

    def gather_bonds(self) -> np.ndarray:
        return self._bonds


class FakeLammps:
    """Minimal stand-in for `lammps.lammps` that can be used to test `LammpsImdForceManager` and `LAMMPSSimulation` without a real LAMMPS build."""

    def __init__(
        self,
        units: str = "real",
        *,
        types: Sequence[int] | None = None,
        masses_by_type: Sequence[float] | None = None,
        rmass: Sequence[float] | None = None,
        unreadable: Iterable[str] = (),
        ids: Sequence[int] | None = None,
        positions: Sequence[Sequence[float]] | None = None,
        box_bounds: Sequence[float] | None = None,
        periodic: Sequence[bool] = (True, True, True),
        bonds: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """
        :param types: per-atom LAMMPS type, 1-based, in global atom order.
        :param masses_by_type: per-type masses. LAMMPS indexes these from 1, so
            element 0 is padding and the array is one longer than `ntypes`.
        :param rmass: per-atom masses, as granular/sphere atom styles carry
            instead of a per-type table.
        :param unreadable: names `extract_atom` should raise on rather than
            return, standing in for a build where the array is absent.
        :param ids: per-atom LAMMPS atom id, in the same global atom order as
            `types`/`positions`. Defaults to 1..N.
        :param positions: per-atom (x, y, z), in the same global atom order.
        :param box_bounds: (xlo, xhi, ylo, yhi, zlo, zhi). Defaults to a large
            box so PBC filtering never trips unless a test asks for it.
        :param periodic: (xperiodic, yperiodic, zperiodic).
        :param bonds: rows of (bond_type, id1, id2), as `gather_bonds` returns them.
        """
        self._units = units
        self._types = np.asarray(types if types is not None else [], dtype=np.int32)
        self._unreadable = frozenset(unreadable)
        # The ctypes buffers must outlive the numpy views taken onto them.
        self._masses_by_type = _as_double_array(masses_by_type)
        self._rmass = _as_double_array(rmass)

        natoms = len(self._types)
        self._ids = np.asarray(ids if ids is not None else range(1, natoms + 1), dtype=np.int64)
        self._positions = np.asarray(
            positions if positions is not None else np.zeros((natoms, 3)), dtype=np.float64
        )
        self._box_bounds = tuple(
            box_bounds if box_bounds is not None else (-1e6, 1e6, -1e6, 1e6, -1e6, 1e6)
        )
        self._periodic = tuple(bool(p) for p in periodic)
        self.numpy = _FakeLammpsNumpy(bonds)

        self.commands: list[str] = []
        self.fix_callbacks: dict[str, Callable] = {}

    def extract_global(self, name: str) -> str | int:
        if name == "units":
            return self._units
        if name == "xperiodic":
            return int(self._periodic[0])
        if name == "yperiodic":
            return int(self._periodic[1])
        if name == "zperiodic":
            return int(self._periodic[2])
        raise KeyError(name)

    def gather_atoms(self, name: str, dtype: int, count: int) -> np.ndarray:
        if name == "type":
            return self._types
        if name == "id":
            return self._ids
        if name == "x":
            return self._positions.reshape(-1)
        raise KeyError(name)

    def extract_atom(self, name: str, dtype: int) -> "ctypes.Array[ctypes.c_double] | None":
        if name in self._unreadable:
            msg = f"cannot read atom array {name!r}"
            raise RuntimeError(msg)
        if name == "mass":
            return self._masses_by_type
        if name == "rmass":
            return self._rmass
        raise KeyError(name)

    def get_natoms(self) -> int:
        return len(self._types)

    def extract_box(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        xlo, xhi, ylo, yhi, zlo, zhi = self._box_bounds
        return (xlo, ylo, zlo), (xhi, yhi, zhi)

    def command(self, command: str) -> None:
        self.commands.append(command)

    def set_fix_external_callback(self, fix_id: str, callback: Callable) -> None:
        self.fix_callbacks[fix_id] = callback


class FakeLammpsNoGlobals:
    """A handle whose `extract_global` always fails."""

    def extract_global(self, name: str) -> str:
        msg = f"no global {name!r}"
        raise RuntimeError(msg)


class FakeLammpsDead:
    """A handle that rejects every command."""

    def command(self, command: str) -> None:
        msg = "LAMMPS instance is closed"
        raise RuntimeError(msg)


class FakeImdState:
    """Stand-in for `ImdStateWrapper`."""

    def __init__(self, interactions: dict | None = None) -> None:
        self.active_interactions = dict(interactions or {})


class FakeFramePublisher:
    """Records every frame sent through it, so a test can inspect what a reset/step actually broadcast."""

    def __init__(self) -> None:
        self.frames: list[Any] = []
        self.clears: int = 0

    def send_frame(self, frame: Any) -> None:
        self.frames.append(frame)

    def send_clear(self) -> None:
        self.clears += 1


class FakeAppServer:
    """Stand-in for the NanoVer app server, exposing only what `LAMMPSSimulation` touches."""

    def __init__(self, imd: Any = None) -> None:
        self.imd = imd
        self.frame_publisher = FakeFramePublisher()
