import pytest

from fakes import FakeLammps, FakeLammpsNoGlobals
from nanover_extensions.lammps.imd import detect_lammps_units, get_unit_conversions


@pytest.mark.parametrize(
    ("style", "expected_pos"),
    [("real", 0.1), ("metal", 0.1), ("si", 1e9), ("nano", 1.0)],
)
def test_position_conversion(style: str, expected_pos: float) -> None:
    pos_to_nm, _, _ = get_unit_conversions(style)
    assert pos_to_nm == expected_pos


def test_real_units_masses_are_already_amu() -> None:
    _pos_to_nm, _force_factor, mass_to_amu = get_unit_conversions("real")
    assert mass_to_amu == 1.0


def test_metal_units_conversions() -> None:
    _pos_to_nm, force_factor, _mass_to_amu = get_unit_conversions("metal")
    assert force_factor == pytest.approx(1.03643e-3)


def test_unknown_units_conversions() -> None:
    with pytest.raises(ValueError, match="Unsupported LAMMPS unit style"):
        get_unit_conversions("apples")


def test_detects_metal_units() -> None:
    sim = FakeLammps("metal")
    assert detect_lammps_units(sim) == "metal"


def test_unknown_units_fall_back_to_real_with_warning() -> None:
    sim = FakeLammps("apples")
    with pytest.warns(UserWarning, match="Unsupported or undetected"):
        assert detect_lammps_units(sim) == "real"


def test_undetectable_units_fall_back_to_real_with_warning() -> None:
    with pytest.warns(UserWarning, match="Could not detect"):
        assert detect_lammps_units(FakeLammpsNoGlobals()) == "real"
