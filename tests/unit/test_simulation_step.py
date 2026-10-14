"""Tests for how `LAMMPSSimulation.step` decides what setup LAMMPS needs before each run."""

import pytest

pytest.importorskip("lammps")

from simulation_helpers import make_simulation

from fakes import FakeAppServer, FakeLammps


@pytest.fixture
def lmp() -> FakeLammps:
    return FakeLammps(types=[1, 1], ids=[10, 20], box_bounds=(0.0, 10.0, 0.0, 10.0, 0.0, 10.0))


def test_the_first_run_sets_up_the_run(lmp: FakeLammps) -> None:
    sim = make_simulation(lmp=lmp)

    sim.step(5)

    assert lmp.commands == ["run 5 post no"]


def test_later_runs_skip_setup_and_teardown(lmp: FakeLammps) -> None:
    sim = make_simulation(lmp=lmp)

    sim.step(5)
    sim.step(5)

    assert lmp.commands[-1] == "run 5 pre no post no"


def test_the_first_run_after_a_reset_sets_up_the_new_fix(lmp: FakeLammps) -> None:
    # reset() registers a fresh `fix external`. LAMMPS only incorporates it on a run that does
    # setup, so a `pre no` run here would leave IMD forces silently un-applied.
    sim = make_simulation(lmp=lmp, generate_bonds=False)
    sim.step(5)  # a run has already happened, so setup would normally be skipped

    sim.reset(FakeAppServer())
    sim.step(5)

    assert lmp.commands[-1] == "run 5 post no"
