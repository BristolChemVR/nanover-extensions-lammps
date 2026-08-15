"""Test doubles standing in for the real LAMMPS Python bindings."""


class FakeLammps:
    """Minimal stand-in for `lammps.lammps` covering the calls we make on the handle."""

    def __init__(self, units: str) -> None:
        self._units = units

    def extract_global(self, name: str) -> str:
        if name != "units":
            raise KeyError(name)
        return self._units


class FakeLammpsNoGlobals:
    """A handle whose `extract_global` always fails, to exercise the fallback path."""

    def extract_global(self, name: str) -> str:
        msg = f"no global {name!r}"
        raise RuntimeError(msg)
