# Matterwave

[**Intro**](#intro) | [**Installation**](#installation) | [**Documentation**](https://qstheory.github.io/matterwave/main/) | [**FFTArray Publication (arXiv)**](https://arxiv.org/abs/2508.03697) | [**Change log**](https://qstheory.github.io/matterwave/main/changelog.html)

This library is a collection of routines and constants to help building simulations in the field of matter wave optics built on top of the [FFTArray](https://github.com/QSTheory/fftarray) library.
See also the [documentation](https://qstheory.github.io/matterwave/main/) of matterwave.

## Intro

The main features of matterwave include:
- A second-order split-step implementation for normal and imaginary time propagation.
- Functions for treating `fftarray.Array` instances as quantum mechanical wave functions, for example normalization, scalar product, expectation values and energy computation.
- Some helpful constants, currently only for the D2 line of Rb87.
- Some plotting helpers for the [panel](https://panel.holoviz.org/) library.

## Installation
For most use cases we recommend installing the optional constraint solver of FFTArray for easy Dimension definition with the `dimsolver` option:
```shell
pip install matterwave fftarray[dimsolver]
```
