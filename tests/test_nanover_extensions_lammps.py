import pytest
import numpy as np
from nanover_extensions.lammps.imd import detect_lammps_units, get_unit_conversions
from nanover_extensions.lammps.converter import lammps_to_frame_data

def test_real_units_conversions() -> None:
    pos_to_nm, force_factor, mass_to_amu = get_unit_conversions("real")
    assert pos_to_nm == 0.1
    assert mass_to_amu == 1.0

def test_unknown_units_conversions() -> None:
    with pytest.raises(ValueError):
        get_unit_conversions("apples")

def test_metal_units_conversions() -> None:
    pos_to_nm, force_factor, mass_to_amu = get_unit_conversions("metal")
    assert force_factor == pytest.approx(1.03643e-3)

def test_box_bounds_become_edge_lengths() -> None:
    frame_data = lammps_to_frame_data(box_bounds_nm=(0.0, 1.0, 0.0, 2.0, 0.0, 3.0))
    np.testing.assert_allclose(np.diag(frame_data.box_vectors), [1.0, 2.0, 3.0])

@pytest.fixture
def two_atom_positions() -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [0, 0, 0.1]])

def test_close_hyrdrogen_bonds(two_atom_positions: np.ndarray) -> None:
    threshold = 1.5
    distances = np.linalg.norm(two_atom_positions[0] - two_atom_positions[1])
    assert distances < threshold

@pytest.mark.parametrize(("style", "expected_pos"), [("real", 0.1), ("metal", 0.1), ("si", 1e9), ("nano", 1.0)])

def test_position_conversion(style: str, expected_pos: float) -> None:
    pos_to_nm, _, _ = get_unit_conversions(style)
    assert pos_to_nm == expected_pos



class FakeLammps:
      def __init__(self, units: str) -> None:
          self._units = units

      def extract_global(self, name: str) -> str:
          if name != "units":
              raise KeyError(name)
          return self._units

def test_detects_metal_units() -> None:
    sim = FakeLammps("metal")
    assert detect_lammps_units(sim) == "metal"