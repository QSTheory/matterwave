
from fftarray.fft_constraint_solver import fft_dim_from_constraints

from matterwave.helpers import plot_fftarray

x_dim = fft_dim_from_constraints(name="x", n=64, pos_middle=0, freq_middle=0, d_pos=1)
y_dim = fft_dim_from_constraints(name="y", n=64, pos_middle=0, freq_middle=0, d_pos=1)
z_dim = fft_dim_from_constraints(name="z", n=64, pos_middle=0, freq_middle=0, d_pos=1)

one_dim_fftarray = x_dim.fft_array(space="pos")
two_dim_fftarray = x_dim.fft_array(space="pos") + y_dim.fft_array(space="pos")
three_dim_fftarray = x_dim.fft_array(space="pos") + y_dim.fft_array(space="pos") + z_dim.fft_array(space="pos")

plot_fftarray(one_dim_fftarray)
plot_fftarray(two_dim_fftarray)
plot_fftarray(three_dim_fftarray)
