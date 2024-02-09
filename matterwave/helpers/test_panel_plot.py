import pytest

from fftarray.backends.tensor_lib import TensorLib
from fftarray.backends.jax_backend import JaxTensorLib
from fftarray.backends.np_backend import NumpyTensorLib
from fftarray.backends.pyfftw_backend import PyFFTWTensorLib
from fftarray.fft_constraint_solver import fft_dim_from_constraints

from matterwave.helpers import plot_fftarray


@pytest.mark.parametrize("tlib", [JaxTensorLib(), NumpyTensorLib(), PyFFTWTensorLib()])
def test_plot_in_dims(tlib: TensorLib):
    x_dim = fft_dim_from_constraints(name="x", n=64, pos_middle=0, freq_middle=0, d_pos=1)
    y_dim = fft_dim_from_constraints(name="y", n=64, pos_middle=0, freq_middle=0, d_pos=1)
    z_dim = fft_dim_from_constraints(name="z", n=64, pos_middle=0, freq_middle=0, d_pos=1)

    one_dim_fftarray = x_dim.fft_array(tlib, space="pos")
    two_dim_fftarray = x_dim.fft_array(tlib, space="pos") + y_dim.fft_array(tlib, space="pos")
    three_dim_fftarray = x_dim.fft_array(tlib, space="pos") + y_dim.fft_array(tlib, space="pos") + z_dim.fft_array(tlib, space="pos")

    # Just test that it does not crash for now...
    plot_fftarray(one_dim_fftarray)
    plot_fftarray(two_dim_fftarray)
    plot_fftarray(three_dim_fftarray)
