# CS336 Spring 2026 Assignment 2: Systems

For a full description of the assignment, see the assignment handout at
[cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

This directory is organized as follows:

- `[./cs336-basics](./cs336-basics)`: directory containing a module
`cs336_basics` and its associated `pyproject.toml`. This module contains the staff
implementation of the language model from assignment 1. If you want to use your own
implementation, you can replace this directory with your own implementation.
- `[./cs336_systems](./cs336_systems)`: This folder is basically empty! This is the
module where you will implement your optimized Transformer language model.
Feel free to take whatever code you need from assignment 1 (in `cs336-basics`) and copy it
over as a starting point. In addition, you will implement distributed training and
optimization in this module.

Visually, it should look something like:

```sh
.
├── cs336_basics  # A python module named cs336_basics
│   ├── __init__.py
│   └── ... other files in the cs336_basics module, taken from assignment 1 ...
├── cs336_systems  # TODO(you): code that you'll write for assignment 2
│   ├── __init__.py
│   └── ... TODO(you): any other files or folders you need for assignment 2 ...
├── README.md
├── pyproject.toml
└── ... TODO(you): other files or folders you need for assignment 2 ...
```

If you would like to use your own implementation of assignment 1, replace the `cs336-basics`
directory with your own implementation, or edit the outer `pyproject.toml` file to point to your
own implementation.

1. We use `uv` to manage dependencies. You can verify that the code from the `cs336-basics`
  package is accessible by running:

```sh
$ uv run python
Using CPython 3.13.13
Creating virtual environment at: /path/to/uv/env/dir
      Built cs336-systems @ file:///path/to/systems/dir
      Built cs336-basics @ file:///path/to/basics/dir
Installed 78 packages in 168ms
Python 3.13.13 (main, Apr  7 2026, 20:49:46) [Clang 22.1.1 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import cs336_basics
...
```

`uv run` installs dependencies automatically as dictated in the `pyproject.toml` file.

## Triton IntelliSense on macOS (editor only)

Triton ships **Linux-only** wheels, so `import triton` / `import triton.language`
cannot be installed (or run) on macOS and the editor reports
`Import "triton.language" could not be resolved`. To get full highlighting,
autocomplete, and hovers for `tl.`* locally, extract the Triton **source** into
`.triton-stubs/` and let Pyright/Pylance resolve from it (it never executes the
Linux binaries — static analysis only):

```sh
uv pip install --no-deps --target .triton-stubs \
  --python-platform linux --python-version 3.12 triton
```

The repo already includes the config that points the language server at it:

- `pyrightconfig.json` -> `"extraPaths": [".triton-stubs"]` (does the resolution)
- `.vscode/settings.json` -> selects the `.venv` interpreter + semantic highlighting

`.triton-stubs/` is git-ignored, so rerun the command above after a fresh clone,
then reload your editor window. Note: this only fixes editing — Triton kernels
still execute only on a Linux GPU machine (e.g. Modal).

## Submitting

To submit, run `./test_and_make_submission.sh` . This script will install your
code's dependencies, run tests, and create a gzipped tarball with the output. We
should be able to unzip your submitted tarball and run
`./test_and_make_submission.sh` to verify your test results.