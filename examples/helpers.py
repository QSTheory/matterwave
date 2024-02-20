from typing import Optional

import numpy as np
from bokeh.plotting import figure, row, show

from fftarray import FFTArray

def plt_wavefunction(
        arr: FFTArray,
        data_name: Optional[str] = None,
        show_plot: bool = True,
    ):
    if len(arr.dims) == 1:
        dim = arr.dims[0]
        p_pos = figure(width=450, height=400, x_axis_label = f"{dim.name} pos coordinate [m]", min_border=50)
        p_pos.line(dim.np_array(space="pos"), np.abs(arr.into(space="pos").values)**2, line_width=2)
        p_pos.title.text = f"{data_name or 'Wavefunction'} propability in position space" # type: ignore

        p_freq = figure(width=450, height=400, x_axis_label = f"{dim.name} freq coordinate [1/m]", min_border=50)
        p_freq.line(dim.np_array(space="freq"), np.abs(arr.into(space="freq").values)**2, line_width=2)
        p_freq.title.text = f"{data_name or 'Wavefunction'} probability in frequency space" # type: ignore

        plot = row([p_pos, p_freq], sizing_mode="stretch_width") # type: ignore
    else:
        raise NotImplementedError

    if show_plot:
        show(plot)
    else:
        return plot