import numpy as np

import pytest

from fftarray.fft_array import FFTDimension, LazyState
from matterwave.split_step import split_step
from fftarray.backends.jax_backend import JaxTensorLib
from fftarray.backends.np_backend import NumpyTensorLib
from fftarray.backends.pyfftw_backend import PyFFTWTensorLib
from fftarray.backends.torch_backend import TorchTensorLib

def test_eager() -> None:
    dim_pos_x = FFTDimension("x", n = 4, d_pos = 1., pos_min = 0.3, freq_min = 0.7, default_eager=False)
    arr = dim_pos_x.pos_array()

    assert split_step(arr, dt=1., mass=1., V=arr)._lazy_state != LazyState()
    arr = arr.set_eager(True)
    assert split_step(arr.evaluate_lazy_state(), dt=1., mass=1., V=arr.evaluate_lazy_state())._lazy_state is None

@pytest.mark.parametrize("tlib", [NumpyTensorLib(), JaxTensorLib(), PyFFTWTensorLib(), TorchTensorLib()])
def test_psi(tlib) -> None:
    dim_pos_x = FFTDimension("x", n = 4, d_pos = 1., pos_min = 0.3, freq_min = 0.7, default_eager=False, default_tlib=tlib)
    # TODO Actually test the result and not just that it does not crash.
    split_step(dim_pos_x.pos_array(), dt=1., mass=1., V=lambda psi: np.abs(psi)**2)