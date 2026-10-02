"""Construct a `LAMMPSSimulation` for tests without going through `__init__`.

`__init__` builds a real `lammps.lammps` handle and loads an input script — it has
no injection point for a fake LAMMPS handle. Every test that exercises instance
methods (as opposed to the `@staticmethod`s) needs to build the instance by hand
instead and set only the attributes the method under test relies on.
"""

from typing import Any

import numpy as np
import pytest

pytest.importorskip("lammps")

from nanover_extensions.lammps.simulation import LAMMPSSimulation


def make_simulation(lmp: Any, **overrides: Any) -> LAMMPSSimulation:
    sim = object.__new__(LAMMPSSimulation)
    defaults: dict[str, Any] = {
        "input_script": "unused",
        "include_velocities": False,
        "include_forces": False,
        "generate_bonds": True,
        "frame_interval": 1,
        "type_to_atomic_number": {},
        "name": "fake",
        "lmp": lmp,
        "lammps_units": "real",
        "_is_periodic": np.array([True, True, True], dtype=bool),
        "_pos_to_nm": 0.1,  # matches "real" units
        "_app_server": None,
        "_current_step": 0,
        "_id_to_index": None,
        "_bond_pairs": None,
        "_bond_orders": None,
        "_particle_elements": None,
        "_imd_force_manager": None,
        "_needs_pre": True,
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(sim, key, value)
    return sim
