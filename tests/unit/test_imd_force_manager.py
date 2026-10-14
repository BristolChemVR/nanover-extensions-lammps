"""Tests for the `fix external` force injection mechanism in `LammpsImdForceManager`."""

from typing import Any

import numpy as np
import pytest
from nanover.imd.imd_force import calculate_imd_force
from nanover.imd.particle_interaction import ParticleInteraction
from nanover.trajectory import FrameData, MissingDataError

from fakes import FakeImdState, FakeLammps, FakeLammpsDead
from nanover_extensions.lammps.imd import LammpsImdForceManager

ID_TO_INDEX = {10: 0, 20: 1, 30: 2}

# Two hydrogens and an oxygen
TYPES = [1, 1, 2]
MASSES_BY_TYPE = [0.0, 1.008, 15.999]  # index 0 is LAMMPS' unused padding slot

PULLED_PARTICLE = 2
PULL_TARGET = (2.0, 0.0, 0.0)


@pytest.fixture
def lmp() -> FakeLammps:
    return FakeLammps("real", types=TYPES, masses_by_type=MASSES_BY_TYPE)


@pytest.fixture
def state() -> FakeImdState:
    """The IMD state a connected client writes its interactions into."""
    return FakeImdState()


@pytest.fixture
def positions() -> np.ndarray:
    """Three atoms strung along the x axis, in nm, in NanoVer index order."""
    return np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])


def make_manager(
    lmp: FakeLammps,
    imd_state: Any = None,
    **kwargs: Any,
) -> LammpsImdForceManager:
    return LammpsImdForceManager(
        lmp=lmp,
        imd_state=imd_state,
        id_to_index=dict(ID_TO_INDEX),
        **kwargs,
    )


def grab(
    state: FakeImdState,
    index: int = PULLED_PARTICLE,
    towards: tuple[float, float, float] = PULL_TARGET,
) -> None:
    """Start pulling particle *index* towards *towards*, as a client grab does."""
    state.active_interactions["grab"] = ParticleInteraction(
        position=towards,
        particles=(index,),
        scale=10.0,
    )


def release(state: FakeImdState) -> None:
    """Let go of every interaction, as a client does when the grab ends."""
    state.active_interactions.clear()


def run_callback(lmp: FakeLammps, tags: list[int], fexternal: np.ndarray) -> None:
    """Invoke the registered fix-external callback the way LAMMPS would."""
    callback = lmp.fix_callbacks[LammpsImdForceManager.FIX_ID]
    callback(lmp, 0, len(tags), tags, (), fexternal)


def test_a_fix_external_is_registered_on_construction(lmp: FakeLammps) -> None:
    make_manager(lmp)
    assert lmp.commands == ["fix imd_nanover all external pf/callback 1 1"]


def test_the_callback_is_registered_against_that_fix(lmp: FakeLammps) -> None:
    manager = make_manager(lmp)
    assert LammpsImdForceManager.FIX_ID in lmp.fix_callbacks
    assert lmp.fix_callbacks[LammpsImdForceManager.FIX_ID].__self__ is manager


def test_unfix_removes_the_fix(lmp: FakeLammps) -> None:
    manager = make_manager(lmp)
    manager.unfix()
    assert lmp.commands[-1] == "unfix imd_nanover"


def test_unfix_tolerates_a_dead_lammps_handle(lmp: FakeLammps) -> None:
    manager = make_manager(lmp)
    manager.lmp = FakeLammpsDead()
    manager.unfix()


def test_forces_reach_the_atom_that_owns_them(
    lmp: FakeLammps,
    state: FakeImdState,
    positions: np.ndarray,
) -> None:
    grab(state)
    manager = make_manager(lmp, state)
    manager.update_interactions(positions)

    # set fexternal to a nonzero value so we can see it get overwritten
    fexternal = np.zeros((3, 3))
    run_callback(lmp, [30, 10, 20], fexternal)

    # id 30 is NanoVer index 2, the particle being pulled, and it is local atom 0.
    assert fexternal[0, 0] > 0.0
    np.testing.assert_array_equal(fexternal[1:], 0.0)


def test_atoms_absent_from_the_map_are_left_alone(
    lmp: FakeLammps,
    state: FakeImdState,
    positions: np.ndarray,
) -> None:
    grab(state)
    manager = make_manager(lmp, state)
    manager.update_interactions(positions)

    fexternal = np.zeros((4, 3))
    run_callback(lmp, [10, 20, 15, 30], fexternal)  # id 15 maps to no NanoVer particle

    np.testing.assert_array_equal(fexternal[2], 0.0)
    assert fexternal[3, 0] > 0.0


def test_stale_values_in_fexternal_are_overwritten(
    lmp: FakeLammps,
    state: FakeImdState,
    positions: np.ndarray,
) -> None:
    grab(state)
    manager = make_manager(lmp, state)
    manager.update_interactions(positions)

    fexternal = np.full((3, 3), 99.0)
    run_callback(lmp, [10, 20, 30], fexternal)

    np.testing.assert_array_equal(fexternal[:2], 0.0)


def test_fexternal_is_zeroed_when_no_interaction_has_ever_been_applied(
    lmp: FakeLammps,
    state: FakeImdState,
) -> None:
    make_manager(lmp, state)
    fexternal = np.full((3, 3), 99.0)
    run_callback(lmp, [10, 20, 30], fexternal)
    np.testing.assert_array_equal(fexternal, 0.0)


def test_releasing_an_interaction_stops_the_applied_force(
    lmp: FakeLammps,
    state: FakeImdState,
    positions: np.ndarray,
) -> None:
    """The user let go: the next timestep must not keep pulling."""
    grab(state)
    manager = make_manager(lmp, state)
    manager.update_interactions(positions)
    run_callback(lmp, [10, 20, 30], np.zeros((3, 3)))

    release(state)
    manager.update_interactions(positions)

    fexternal = np.full((3, 3), 99.0)
    run_callback(lmp, [10, 20, 30], fexternal)
    np.testing.assert_array_equal(fexternal, 0.0)


def test_releasing_an_interaction_clears_the_broadcast_force_and_energy(
    lmp: FakeLammps,
    state: FakeImdState,
    positions: np.ndarray,
) -> None:
    grab(state)
    manager = make_manager(lmp, state)
    manager.update_interactions(positions)

    release(state)
    manager.update_interactions(positions)

    np.testing.assert_array_equal(manager.user_forces, np.zeros((3, 3)))
    assert manager.total_user_energy == 0.0


def test_updating_with_nothing_grabbed_is_a_no_op(
    lmp: FakeLammps,
    state: FakeImdState,
    positions: np.ndarray,
) -> None:
    """A client is connected but nobody is dragging anything."""
    manager = make_manager(lmp, state)
    manager.update_interactions(positions)

    assert manager.total_user_energy == 0.0
    assert manager.user_forces.size == 0


def test_without_an_imd_state_no_force_is_ever_applied(
    lmp: FakeLammps,
    positions: np.ndarray,
) -> None:
    """A simulation loaded outside of IMD should not crash, and should not apply any forces."""
    manager = make_manager(lmp, None)
    manager.update_interactions(positions)

    fexternal = np.full((3, 3), 99.0)
    run_callback(lmp, [10, 20, 30], fexternal)
    np.testing.assert_array_equal(fexternal, 0.0)


def test_a_tag_above_the_largest_mapped_id_raises(
    lmp: FakeLammps,
    state: FakeImdState,
    positions: np.ndarray,
) -> None:
    """The callback is given a tag that is not in the ID map, which is an error."""
    grab(state)
    manager = make_manager(lmp, state)
    manager.update_interactions(positions)

    with pytest.raises(IndexError):
        run_callback(lmp, [10, 20, 40], np.zeros((3, 3)))


# --- unit conversion -----


@pytest.mark.parametrize(
    ("style", "force_factor"),
    [("real", 0.023901), ("metal", 1.03643e-3), ("nano", 0.069477)],
)
def test_applied_forces_are_converted_into_lammps_units(
    state: FakeImdState,
    positions: np.ndarray,
    style: str,
    force_factor: float,
) -> None:
    lmp = FakeLammps(style, types=TYPES, masses_by_type=MASSES_BY_TYPE)
    grab(state)
    manager = make_manager(lmp, state)
    manager.update_interactions(positions)

    fexternal = np.zeros((3, 3))
    run_callback(lmp, [10, 20, 30], fexternal)

    np.testing.assert_allclose(fexternal, manager.user_forces * force_factor, rtol=1e-6)


def test_broadcast_forces_stay_in_nanover_units(
    lmp: FakeLammps,
    state: FakeImdState,
    positions: np.ndarray,
) -> None:
    """`user_forces` is what the client sees, so it must not carry LAMMPS scaling."""
    grab(state)
    manager = make_manager(lmp, state)
    manager.update_interactions(positions)

    _energy, unscaled = calculate_imd_force(
        positions,
        np.array(MASSES_BY_TYPE)[TYPES],
        state.active_interactions.values(),
    )
    np.testing.assert_allclose(manager.user_forces, unscaled, rtol=1e-6)


def test_the_unit_style_is_detected_when_it_is_not_given(
    state: FakeImdState,
    positions: np.ndarray,
) -> None:
    lmp = FakeLammps("metal", types=TYPES, masses_by_type=MASSES_BY_TYPE)
    grab(state)
    manager = make_manager(lmp, state)
    manager.update_interactions(positions)

    fexternal = np.zeros((3, 3))
    run_callback(lmp, [10, 20, 30], fexternal)
    np.testing.assert_allclose(fexternal, manager.user_forces * 1.03643e-3, rtol=1e-6)


def test_an_explicit_unit_style_overrides_detection(
    state: FakeImdState,
    positions: np.ndarray,
) -> None:
    lmp = FakeLammps("metal", types=TYPES, masses_by_type=MASSES_BY_TYPE)
    grab(state)
    manager = make_manager(lmp, state, lammps_units="real")
    manager.update_interactions(positions)

    fexternal = np.zeros((3, 3))
    run_callback(lmp, [10, 20, 30], fexternal)
    np.testing.assert_allclose(fexternal, manager.user_forces * 0.023901, rtol=1e-6)


# ---massess------


def test_per_type_masses_are_expanded_over_atoms(lmp: FakeLammps) -> None:
    manager = make_manager(lmp)
    np.testing.assert_allclose(manager._masses, [1.008, 1.008, 15.999])


def test_masses_are_converted_to_amu() -> None:
    lmp = FakeLammps("nano", types=[1, 1, 1], masses_by_type=[0.0, 2.0])
    manager = make_manager(lmp)
    np.testing.assert_allclose(manager._masses, 2.0 * 6.02214076e5)


def test_per_atom_rmass_is_used_when_there_is_no_per_type_table() -> None:
    """Granular and sphere atom styles carry a mass per atom instead of per type."""
    lmp = FakeLammps("real", types=[1, 1, 1], rmass=[1.5, 2.5, 3.5])
    manager = make_manager(lmp)
    np.testing.assert_allclose(manager._masses, [1.5, 2.5, 3.5])


def test_unreadable_per_type_masses_warn_and_fall_back_to_rmass() -> None:
    lmp = FakeLammps(
        "real",
        types=TYPES,
        masses_by_type=MASSES_BY_TYPE,
        rmass=[1.5, 2.5, 3.5],
        unreadable=["mass"],
    )
    with pytest.warns(UserWarning, match="trying per-atom rmass"):
        manager = make_manager(lmp)
    np.testing.assert_allclose(manager._masses, [1.5, 2.5, 3.5])


def test_every_step_of_the_mass_fallback_is_reported() -> None:
    """A handle that can read neither table should say so twice, then give up."""
    lmp = FakeLammps(
        "real",
        types=TYPES,
        masses_by_type=MASSES_BY_TYPE,
        rmass=[1.5, 2.5, 3.5],
        unreadable=["mass", "rmass"],
    )
    with pytest.warns(UserWarning, match="mass") as raised:
        manager = make_manager(lmp)

    reported = [str(warning.message) for warning in raised]
    assert any("trying per-atom rmass" in message for message in reported)
    assert any("Could not read per-atom rmass" in message for message in reported)
    assert any("using unit masses" in message for message in reported)
    np.testing.assert_allclose(manager._masses, [1.0, 1.0, 1.0])


def test_missing_masses_fall_back_to_unit_masses_with_a_warning() -> None:
    lmp = FakeLammps("real", types=TYPES)
    with pytest.warns(UserWarning, match="using unit masses"):
        manager = make_manager(lmp)
    np.testing.assert_allclose(manager._masses, [1.0, 1.0, 1.0])


def test_orthorhombic_box_lengths_are_taken_from_the_diagonal(lmp: FakeLammps) -> None:
    manager = make_manager(lmp, pbc_vectors=np.diag([1.0, 2.0, 3.0]))
    assert manager.periodic_box_lengths is not None
    np.testing.assert_allclose(manager.periodic_box_lengths, [1.0, 2.0, 3.0])


def test_without_a_box_there_is_no_minimum_image_convention(lmp: FakeLammps) -> None:
    manager = make_manager(lmp)
    assert manager.periodic_box_lengths is None


def test_a_triclinic_box_warns_that_only_orthorhombic_is_supported(lmp: FakeLammps) -> None:
    tilted = np.array([[1.0, 0.0, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 3.0]])
    with pytest.warns(UserWarning, match="orthorhombic"):
        manager = make_manager(lmp, pbc_vectors=tilted)
    assert manager.periodic_box_lengths is not None
    np.testing.assert_allclose(manager.periodic_box_lengths, [1.0, 2.0, 3.0])


def test_user_energy_is_written_to_the_frame(
    lmp: FakeLammps,
    state: FakeImdState,
    positions: np.ndarray,
) -> None:
    grab(state)
    manager = make_manager(lmp, state)
    manager.update_interactions(positions)

    frame = FrameData()
    manager.add_to_frame_data(frame)
    assert frame.user_energy == pytest.approx(manager.total_user_energy)


def test_only_interacting_particles_appear_in_the_frame(
    lmp: FakeLammps,
    state: FakeImdState,
    positions: np.ndarray,
) -> None:
    grab(state)
    manager = make_manager(lmp, state)
    manager.update_interactions(positions)

    frame = FrameData()
    manager.add_to_frame_data(frame)

    np.testing.assert_array_equal(frame.user_forces_index, [PULLED_PARTICLE])
    np.testing.assert_allclose(
        frame.user_forces_sparse,
        [manager.user_forces[PULLED_PARTICLE]],
        rtol=1e-6,
    )


def test_a_frame_carries_no_forces_before_any_interaction(lmp: FakeLammps) -> None:
    manager = make_manager(lmp)

    frame = FrameData()
    manager.add_to_frame_data(frame)

    assert frame.user_energy == 0.0
    with pytest.raises(MissingDataError):
        _ = frame.user_forces_sparse
