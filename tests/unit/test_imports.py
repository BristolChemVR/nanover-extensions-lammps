import importlib
import pkgutil

import pytest
from typer.testing import CliRunner

import nanover_extensions.lammps

MODULES = [
    m.name
    for m in pkgutil.walk_packages(
        nanover_extensions.lammps.__path__, prefix="nanover_extensions.lammps."
    )
]


@pytest.mark.parametrize("module_name", MODULES)
def test_import_module(module_name: str) -> None:
    """Test that a module can be imported without errors."""
    importlib.import_module(module_name)


def test_cli_help() -> None:
    from nanover_extensions.lammps.cli.main import app

    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
