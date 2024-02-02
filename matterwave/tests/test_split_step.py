import numpy as np

import pytest

from fftarray.fft_array import FFTDimension
from matterwave.split_step import split_step
from fftarray.backends.jax_backend import JaxTensorLib
from fftarray.backends.np_backend import NumpyTensorLib
from fftarray.backends.pyfftw_backend import PyFFTWTensorLib

def test_eager() -> None:
    dim_pos_x = FFTDimension("x", n = 4, d_pos = 1., pos_min = 0.3, freq_min = 0.7, default_eager=False)
    arr = dim_pos_x.fft_array(space="pos")

    # TODO: Needs lazy implemented to work
    # assert split_step(arr, dt=1., mass=1., V=arr)._factors_applied == (False,)
    arr = arr.into(eager=True)
    assert split_step(arr.into(factors_applied=True), dt=1., mass=1., V=arr.into(factors_applied=True))._factors_applied == (True,)

@pytest.mark.parametrize("tlib", [NumpyTensorLib(), JaxTensorLib(), PyFFTWTensorLib()])
def test_psi(tlib) -> None:
    dim_pos_x = FFTDimension("x", n = 4, d_pos = 1., pos_min = 0.3, freq_min = 0.7, default_eager=False, default_tlib=tlib)
    # TODO Actually test the result and not just that it does not crash.
    split_step(dim_pos_x.fft_array(space="pos"), dt=1., mass=1., V=lambda psi: np.abs(psi)**2)