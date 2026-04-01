import typer

from nanover_extensions.cli.lammps_cli import app as lammps_app

app = typer.Typer()

app.add_typer(lammps_app)

if __name__ == "__main__":
    app()
