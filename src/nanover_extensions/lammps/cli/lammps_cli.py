import os
import time

from glob import glob
from pathlib import Path

from typing import Annotated

import typer

from nanover.app.omni import OmniRunner

os.environ["OMP_NUM_THREADS"] = "4"

from nanover.omni import OmniRunner
from nanover.websocket.record import record_from_runner

# try:
import nanover_extensions.lammps.simulation
from nanover_extensions.lammps.simulation import LAMMPSSimulation

app = typer.Typer()


def _list_to_str(s: str, seperator=",") -> list[str]:
    """Takes a given input of `seperator` split list of files converts to a list of str."""
    return s.split(seperator)


@app.command()
def lammps(
    entries: Annotated[
        list[str],
        typer.Option(
            "-e",
            "--entries",
            help="Simulation(s) to run via LAMMPS (data file format)",
            parser=_list_to_str,
        ),
    ],
    record_to_path: Annotated[
        Path | None,
        typer.Option(
            "-r", "--record-to-path", help="Record trajectory and state to files."
        ),
    ] = None,
    omp_num_threads: Annotated[
        int,
        typer.Option(
            "-nt",
            "--omp-num-threads",
            help="Set OMP_NUM_THREADS for OpenMP parallelism (default: 4).",
        ),
    ] = 4,
    quiet: Annotated[bool, typer.Option(help="Whether to suppress LAMMPS outputs.")] = False,
):
    if omp_num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

    with OmniRunner.with_basic_server() as runner:
        for entry in entries:
            if LAMMPSSimulation is None:
                print(
                    "Skipping --lammps entry: LAMMPS module failed to import (see error above)."
                )
                continue
            # Entry is a list of tokens from nargs="+".
            # Optional trailing integer is the frame interval: --lammps sim.in 20
            *path_tokens, last = entry
            if last.isdigit():
                paths = path_tokens
                frame_interval = int(last)
            else:
                paths = entry
                frame_interval = 1
            for pattern in paths:
                for path in glob(pattern, recursive=True) or [pattern]:
                    try:
                        simulation = LAMMPSSimulation(
                            input_script=path,
                            frame_interval_steps=frame_interval,
                            quiet=quiet,
                        )
                        runner.add_simulation(simulation)
                        natoms = int(simulation.lmp.get_natoms())
                        print(
                            f"LAMMPS simulation with {natoms} atoms loaded from {path} "
                            f"(frame interval: {frame_interval})"
                        )
                    except NotImplementedError as e:
                        print(f"LAMMPS simulation not yet implemented: {e}")
                    except Exception as e:
                        print(f"Error initializing LAMMPS simulation from {path}: {e}")

        runner.print_basic_info()

        # Start the first simulation if available
        if runner.simulations:
            runner.load(0)

        if record_to_path is not None:
            stem = record_to_path
            if stem == "":
                timestamp = time.strftime("%Y-%m-%d-%H%M-%S", time.gmtime())
                stem = f"omni-recording-{timestamp}"

            out_path = f"{stem}.nanover.zip"
            print(f"Recording to {out_path}")

            with record_from_runner(runner, out_path):
                input("Press Enter to stop.\n")
        else:
            input("Press Enter to stop.\n")
