from typing import Any

import pytest
import fftarray as fa

from matterwave import plot_array
from tests.helpers import XPS, PrecisionSpec, precisions

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("precision", precisions)
def test_plot_in_dims(xp: Any, precision: PrecisionSpec):
    x_dim = fa.dim_from_constraints(name="x", n=64, pos_middle=0, freq_middle=0, d_pos=1)
    y_dim = fa.dim_from_constraints(name="y", n=64, pos_middle=0, freq_middle=0, d_pos=1)
    z_dim = fa.dim_from_constraints(name="z", n=64, pos_middle=0, freq_middle=0, d_pos=1)

    x = fa.coords_from_dim(x_dim, "pos", xp=xp, dtype=getattr(xp, precision))
    y = fa.coords_from_dim(y_dim, "pos", xp=xp, dtype=getattr(xp, precision))
    z = fa.coords_from_dim(z_dim, "pos", xp=xp, dtype=getattr(xp, precision))

    one_dim_fftarray = x
    two_dim_fftarray = x+y
    three_dim_fftarray = x+y+z

    # Just test that it does not crash for now...
    plot_array(one_dim_fftarray)
    plot_array(two_dim_fftarray)
    plot_array(three_dim_fftarray)
