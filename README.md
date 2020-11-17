# Light-curve processing library

# Rust

### `light-curve`
WIP: will be an umbrella crate for other Rust crates

### `light-curve-feature`

![docs.rs badge](https://docs.rs/light-curve-feature/badge.svg)

A collection of features to be extracted from light curves

### `light-curve-common`
Common utilities

### `light-curve-interpol`
WIP: interpolation utilities, only linear interpolation is available now

# Python

### `light-curve-python`

[![PyPI version](https://badge.fury.io/py/light-curve-python.svg)](https://pypi.python.org/pypi/light-curve-python/)

Python bindings for `light-curve-feature` Rust crate. Install it by

```sh
pip3 install light-curve-python
```

### `python`

WIP: future home for mixed Rust-Python package. Now pure Python implementation is developed here

# How to develop

### Formatting

All Rust code should be formatted by `rustfmt` and should pass `clippy` check.

All Python code should be formatted by `black` with 120 char line-width.

[`pre-commit`](https://pre-commit.com/) can be used for automatically formatting on `git commit`.
Install Git hooks in the project root via
```shell
pip3 install pre-commit
# OR: pip3 install --user pre-commit
# OR on macOS: brew install pre-commit
pre-commit install
```
