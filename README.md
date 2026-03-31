# :rocket: nanover-extensions-lammps

LAMMPS engine interface for NanoVer

## Setup Dev Environment

Installation is using [UV](https://docs.astral.sh/uv/) to manage everything.

**Step 1**: Create a virtual environment

```
uv venv
```

**Step 2**: Activate your new environment

```
# on windows
.venv\Scripts\activate

# on mac / linux
source .venv/bin/activate
```

**Step 3**: Install all the cool dependencies

```
uv sync
```

## Github Repo Setup

To add your new project to its Github repository, firstly make sure you have created a project named **nanover-extensions-lammps** on Github.
Follow these steps to push your new project.

```
git remote add origin git@github.com:BristolChemVR/nanover-extensions-lammps.git
git branch -M main
git push -u origin main
```

## Built-in CLI Commands

We've included a bunch of useful CLI commands for common project tasks using [taskipy](https://github.com/taskipy/taskipy).

```
# run src/nanover_extensions_lammps/nanover_extensions_lammps.py
task run

# run all tests
task tests



# run test coverage and generate report
task coverage

# typechecking with Ty or Mypy
task type

# ruff linting
task lint

# format with ruff
task format
```

## Docs Generation + Publishing

Doc generation is setup to scan everything inside `/src`, files with a prefix `_` will be ignored. Basic doc functions for generating, serving, and publishing can be done through these CLI commands:

```
# generate docs & serve
task docs

# serve docs
task serve

# generate static HTML docs (outputs to ./site/)
task html

# publish docs to Github Pages
task publish
```

Note: Your repo must be public or have an upgraded account to deploy docs to Github Pages.

## References

- [Pattern](https://github.com/wyattferguson/pattern) - A modern cookiecutter template for your next Python project.

## License

MIT

## Contact

Created by [Thomas Regan](https://github.com/BristolChemVR)
