# Installation

There are different versions of matterwave available for installation, enabling different capabilities and thus, coming with different external packages as requirements.
The bare version features the core capabilities. For example, there is no helper methods to define a `fftarray.Dimension`. There is also no automatic installation of required packages for accelerated FFT implementations on GPUs (`jax`). Additionally, there is a version to enable the execution of the examples.

You can install each version of matterwave from the GitHub repository directly via SSH (recommended) or HTTPS.
```shell
## Bare installation
python -m pip install 'matterwave @ git+ssh://git@github.com/QSTheory/matterwave.git' # SSH
python -m pip install 'matterwave @ git+https://github.com/QSTheory/matterwave.git' # HTTPS
```
**Available versions**
```shell
## JAX support (GPU acceleration)
python -m pip install 'matterwave[jax] @ git+ssh://git@github.com/QSTheory/matterwave.git' # SSH
## Some helper methods (e.g. FFT constraint solver)
python -m pip install 'matterwave[helpers] @ git+ssh://git@github.com/QSTheory/matterwave.git' # SSH
## Examples
python -m pip install 'matterwave[examples] @ git+ssh://git@github.com/QSTheory/matterwave.git' # SSH
```
You can also combine different additions:
```shell
## JAX support + helper methods
python -m pip install 'matterwave[helpers,jax] @ git+ssh://git@github.com/QSTheory/matterwave.git' # SSH
```
