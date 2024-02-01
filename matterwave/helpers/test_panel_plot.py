

from fftarray import FFTDimension

from matterwave.helpers import plot_fftarray

x_dim = FFTDimension(name="x", n=64, pos_middle=0, freq_middle=0, d_pos=1)
y_dim = FFTDimension(name="y", n=64, pos_middle=0, freq_middle=0, d_pos=1)
z_dim = FFTDimension(name="z", n=64, pos_middle=0, freq_middle=0, d_pos=1)

one_dim_fftarray = x_dim.pos_array()
two_dim_fftarray = x_dim.pos_array() + y_dim.pos_array()
three_dim_fftarray = x_dim.pos_array() + y_dim.pos_array() + z_dim.pos_array()

plot_fftarray(one_dim_fftarray)
plot_fftarray(two_dim_fftarray)
plot_fftarray(three_dim_fftarray)
