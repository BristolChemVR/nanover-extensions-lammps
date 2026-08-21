"""Test doubles standing in for the real LAMMPS Python bindings."""

import ctypes
from collections.abc import Callable, Iterable, Sequence

import numpy as np


def _as_double_array(values: Sequence[float] | None) -> "ctypes.Array[ctypes.c_double] | None":
    """Convert python list into ctypes array that lammps can read."""
    if values is None:
        return None
    return (ctypes.c_double * len(values))(*(float(v) for v in values))


class FakeLammps:
    """Minimal stand-in for `lammps.lammps` that can be used to test `LammpsImdForceManager` without a real LAMMPS build."""

    def __init__(
        self,
        units: str = "real",
        *,
        types: Sequence[int] | None = None,
        masses_by_type: Sequence[float] | None = None,
        rmass: Sequence[float] | None = None,
        unreadable: Iterable[str] = (),
    ) -> None:
        """
        :param types: per-atom LAMMPS type, 1-based, in global atom order.
        :param masses_by_type: per-type masses. LAMMPS indexes these from 1, so
            element 0 is padding and the array is one longer than `ntypes`.
        :param rmass: per-atom masses, as granular/sphere atom styles carry
            instead of a per-type table.
        :param unreadable: names `extract_atom` should raise on rather than
            return, standing in for a build where the array is absent.
        """
        self._units = units
        self._types = np.asarray(types if types is not None else [], dtype=np.int32)
        self._unreadable = frozenset(unreadable)
        # The ctypes buffers must outlive the numpy views taken onto them.
        self._masses_by_type = _as_double_array(masses_by_type)
        self._rmass = _as_double_array(rmass)

        self.commands: list[str] = []
        self.fix_callbacks: dict[str, Callable] = {}

    def extract_global(self, name: str) -> str:
        if name != "units":
            raise KeyError(name)
        return self._units

    def gather_atoms(self, name: str, dtype: int, count: int) -> np.ndarray:
        if name != "type":
            raise KeyError(name)
        return self._types

    def extract_atom(self, name: str, dtype: int) -> "ctypes.Array[ctypes.c_double] | None":
        if name in self._unreadable:
            msg = f"cannot read atom array {name!r}"
            raise RuntimeError(msg)
        if name == "mass":
            return self._masses_by_type
        if name == "rmass":
            return self._rmass
        raise KeyError(name)

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
