import click

from nanover_extensions.cli.lammps_cli import lammps


@click.group()
def cli() -> None:
    pass


cli.add_command(lammps)


if __name__ == "__main__":
    cli()
