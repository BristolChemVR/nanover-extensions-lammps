import os
import time

from glob import glob
from pathlib import Path

from typing import Annotated

import typer

from nanover.app.omni import OmniRunner

# ---------------------------------------------------------------------------
# Windows DLL pre-loading
#
# LAMMPS (MSMPI build) is compiled with MinGW-GCC and ships its own copies of
# libgomp-1.dll, libgcc_s_seh-1.dll, libstdc++-6.dll etc.  The conda
# environment also contains different versions of these same DLLs (in
# Library/bin).  If any other nanover dependency (e.g. OpenMM) loads the conda
# versions first, liblammps.dll will fail to load with WinError 127 because
# the already-resident DLL is the wrong version.
#
# Fix: register the LAMMPS bin and MS-MPI bin via os.add_dll_directory() and
# pre-load the GCC runtime DLLs from the LAMMPS bin *before* importing OpenMM
# or any other conda package that might pull in conflicting versions.
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "4"


# _lammps_dll_path = os.environ.get("LAMMPSDLLPATH") or os.environ.get("LAMMPSHOME")
# if _lammps_dll_path:
#     _lammps_bin = (
#         _lammps_dll_path
#         if os.path.isfile(os.path.join(_lammps_dll_path, "liblammps.dll"))
#         else os.path.join(_lammps_dll_path, "bin")
#     )
#     if hasattr(os, "add_dll_directory") and os.path.isdir(_lammps_bin):
#         os.add_dll_directory(_lammps_bin)
#     # Pre-load GCC runtime DLLs from the LAMMPS bin so they take precedence
#     # over any conda-env copies that OpenMM or numpy might otherwise load first.
#     for _dll_name in (
#         "libgomp-1.dll",
#         "libgcc_s_seh-1.dll",
#         "libstdc++-6.dll",
#         "libwinpthread-1.dll",
#     ):
#         _dll_full = os.path.join(_lammps_bin, _dll_name)
#         if os.path.isfile(_dll_full):
#             try:
#                 ctypes.CDLL(_dll_full)
#             except OSError:
#                 pass

# _msmpi_bin = os.environ.get("MSMPI_BIN") or r"C:\Program Files\Microsoft MPI\Bin"
# if hasattr(os, "add_dll_directory") and os.path.isdir(_msmpi_bin):
#     os.add_dll_directory(_msmpi_bin)

from nanover.omni import OmniRunner
from nanover.websocket.record import record_from_runner

# try:
import nanover_extensions.lammps.simulation
from nanover_extensions.lammps.simulation import LAMMPSSimulation

# except Exception as e:
#     print(f"Could not import LAMMPS module: {e}")
#     LAMMPSSimulation = None


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
