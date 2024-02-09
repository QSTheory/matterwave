
from typing import List, Optional

import panel as pn
import pandas as pd # type: ignore
import xarray as xr
import colorcet as cc # type: ignore
import hvplot.xarray  # type: ignore # noqa
from holoviews import streams # type: ignore
import holoviews as hv
from holoviews.operation.datashader import rasterize # type: ignore
pn.extension()

# TODO: check this out when finalising plotting routine
from bokeh.core.validation import silence
from bokeh.core.validation.warnings import FIXED_SIZING_MODE
silence(FIXED_SIZING_MODE, True)

from fftarray import FFTArray, FFTDimension

import numpy as np

COLORS = ['#5BFF00', '#6400E6', '#FF0000'] # green, purple, red
CONTOUR_MAP = cc.CET_CBD2[::-1]


def plot_fftarray(
    fftarray: FFTArray,
) -> None:

    plot = generate_panel_plot(fftarray)

    plot.servable()

def generate_panel_plot(
    fftarray: FFTArray,
) -> pn.Column:

    if len(fftarray.dims) == 0:
        return pn.Column()

    # Create pandas dataframe to plot FFTWave infos in a table
    df_dict = {"dim": [dim for dim in fftarray.dims_dict.keys()]}
    for info in ["n", "d_pos", "d_freq", "pos_min", "pos_max", "freq_min", "freq_max"]:
        df_dict[info] = [float(getattr(fftarray.dims_dict[dim], info)) for dim in df_dict["dim"]]
    value_specs = pd.DataFrame(df_dict)

    dims = [str(dim) for dim in df_dict["dim"]]
    k_dims = [f"k{dim}" for dim in dims]

    # TODO: Why are they not coerced to arrays by xr?
    # Maybe because of the non-standard dims attribute?
    xr_pos = xr.Dataset(
        data_vars={
            "|Psi({0})|^2".format(",".join(dims)): (dims, np.array(np.abs(fftarray.into(space="pos"))**2))
        },
        coords={dim: fftarray.dims_dict[dim].np_array(space="pos") for dim in dims}
    )

    from matterwave.rb87 import hbar, k_L
    xr_freq = xr.Dataset(
        data_vars={
            "|Psi({0})|^2".format(",".join(k_dims)): (k_dims, np.array(np.abs(fftarray.into(space="freq"))**2))
        },
        coords={kdim: fftarray.dims_dict[dim].np_array(space="freq")/k_L*2*np.pi for dim, kdim in zip(dims,k_dims)}
    )

    if len(dims) == 1:
        # TODO Enable datashading if there are too many points
        # TODO Allow customized log-plotting
        return pn.Column(
            pn.pane.DataFrame(value_specs, index = False, width = 800),
            xr_pos.hvplot(kind = "line", title = "Position Space"),
            xr_freq.hvplot(kind = "line", title = "Frequency Space"),
        )

    else:

        def interactive_plots(data_array: xr.Dataset) -> pn.Column:

            # Different variables for position and frequency space
            dims: List[str] = sorted(list(data_array.coords._names)) # type: ignore
            in_frequency_space = 'k' in dims[0]
            center_attribute_name = 'freq_middle' if in_frequency_space else 'pos_middle'
            title = "Frequency space" if in_frequency_space else "Position space"
            name = "|Psi({0})|^2".format(",".join(k_dims)) if in_frequency_space else "|Psi({0})|^2".format(",".join(dims))

            def get_fftarray_dim(
                    fftarray: FFTArray,
                    dim: str
                ) -> FFTDimension:
                dim_name = dim.replace("k", "")
                return fftarray.dims_dict[dim_name]

            # Stream that enables choosing points by clicking within the contour plot
            pointer_stream = streams.SingleTap(
                x=getattr(get_fftarray_dim(fftarray, dims[1]), center_attribute_name),
                y=getattr(get_fftarray_dim(fftarray, dims[0]), center_attribute_name),
            )

            # Creates a contour plot and 2 and 3 line plots for the 2d and 3d case, respectively
            def contour_and_line_plots(free_axis: str, free_axis_val: Optional[float]):

                is_3_dim = free_axis != ''

                if is_3_dim:
                    contour_plot = getattr(data_array, name).sel(indexers={free_axis: free_axis_val}).hvplot(cmap = CONTOUR_MAP).opts(title=f'{free_axis} = {free_axis_val:.3g}')
                else:
                    contour_plot = data_array.hvplot(cmap = CONTOUR_MAP)
                contour_plot.opts(frame_height=400, frame_width=400, colorbar_opts={'title': name})

                nonlocal pointer_stream
                pointer_stream = streams.SingleTap(x=pointer_stream.x, y=pointer_stream.y, source=contour_plot)

                contour_dims = dims.copy()
                if is_3_dim:
                    contour_dims.remove(free_axis)

                def callback_contour_markers(x,y):
                    contour_markers = hv.HLine(y).opts(color=COLORS[0]) * hv.VLine(x).opts(color=COLORS[1])
                    if is_3_dim:
                        contour_markers *= hv.Points((x,y)).opts(color=COLORS[2], marker='x', size=25, line_width=4)
                    return contour_markers

                contour_markers = hv.DynamicMap(callback_contour_markers, streams=[pointer_stream])

                def callback_contour_line_plot_x(x,y):
                    nearest_y = float(data_array.coords[contour_dims[0]].sel(indexers={contour_dims[0]: y}, method="nearest"))
                    ylabel = f"$${name.replace(contour_dims[0], '').replace(free_axis, '').replace(',', '').replace('Psi', f'{chr(92)}Psi')}$$"
                    indexers = {contour_dims[0]: y}
                    title = f"{contour_dims[0]} = {nearest_y:.3g}"
                    if is_3_dim:
                        indexers[free_axis] = free_axis_val
                        title += f", {free_axis} = {free_axis_val:.3g}"
                    data = getattr(data_array, name).sel(indexers=indexers, method='nearest')
                    line_plot_x = data.hvplot().opts(color=COLORS[0], title=title, ylabel=ylabel)

                    return line_plot_x

                def callback_contour_line_plot_y(x,y):
                    nearest_x = float(data_array.coords[contour_dims[1]].sel(indexers={contour_dims[1]: x}, method="nearest"))
                    ylabel = f"$${name.replace(contour_dims[1], '').replace(free_axis, '').replace(',', '').replace('Psi', f'{chr(92)}Psi')}$$"
                    indexers = {contour_dims[1]: x}
                    title = f"{contour_dims[1]} = {nearest_x:.3g}"
                    if is_3_dim:
                        indexers[free_axis] = free_axis_val
                        title += f", {free_axis} = {free_axis_val:.3g}"
                    data = getattr(data_array, name).sel(indexers=indexers, method='nearest')
                    line_plot_y = data.hvplot().opts(color=COLORS[1], title=title, ylabel=ylabel)

                    return line_plot_y

                contour_line_plot_x = hv.DynamicMap(callback_contour_line_plot_x, streams=[pointer_stream])
                contour_line_plot_y = hv.DynamicMap(callback_contour_line_plot_y, streams=[pointer_stream])

                if is_3_dim:
                    def callback(x,y):
                        nearest_x = float(data_array.coords[contour_dims[1]].sel(indexers={contour_dims[1]: x}, method="nearest"))
                        nearest_y = float(data_array.coords[contour_dims[0]].sel(indexers={contour_dims[0]: y}, method="nearest"))
                        vline = hv.VLine(free_axis_val).opts(color='#D3D3D3')
                        ylabel = f"$${name.replace(contour_dims[0], '').replace(contour_dims[1], '').replace(',', '').replace('Psi', f'{chr(92)}Psi')}$$"
                        data = getattr(data_array, name).sel(indexers={contour_dims[0]: y, contour_dims[1]: x}, method='nearest')
                        line_plot = data.hvplot().opts(color=COLORS[2]) * vline
                        line_plot.opts(frame_width=400, title=f'{contour_dims[0]} = {nearest_y:.3g}, {contour_dims[1]} = {nearest_x:.3g}', ylabel=ylabel)

                        return line_plot

                    line_plot = hv.DynamicMap(callback, streams=[pointer_stream])

                    return rasterize(contour_plot) * contour_markers, line_plot, contour_line_plot_x, contour_line_plot_y

                return rasterize(contour_plot) * contour_markers, contour_line_plot_x, contour_line_plot_y

            # Widgets to control the third dimension (fixed for contour plot) and its value
            dim_widget = pn.widgets.Select(name='Controlled dimension', value=dims[-1], options=dims)
            def make_val_widget(free_dim: str):
                return pn.widgets.DiscreteSlider(
                    name=f'{free_dim} value',
                    options=list(data_array.coords[free_dim].values),
                    value = float(getattr(get_fftarray_dim(fftarray, free_dim), center_attribute_name))
                )
            val_widget = make_val_widget(dim_widget.value)

            if len(dims) == 2:
                return pn.Column(
                    f'## {title}',
                    pn.Column(*contour_and_line_plots('', None))
                )

            # 3 Dimensions
            column = pn.Column(
                f'## {title}',
                pn.Row(dim_widget, val_widget, width=400),
                pn.Column(*contour_and_line_plots(dim_widget.value, val_widget.value))
            )

            def update_dim(event):
                contour_dims = dims.copy()
                contour_dims.remove(dim_widget.value)
                pointer_stream.update(x=float(getattr(get_fftarray_dim(fftarray, contour_dims[1]), center_attribute_name)))
                pointer_stream.update(y=float(getattr(get_fftarray_dim(fftarray, contour_dims[0]), center_attribute_name)))

                nonlocal val_widget
                val_widget = make_val_widget(dim_widget.value)
                val_widget.param.watch(update_val, 'value')
                column[1][1] = val_widget
                column[2] = pn.Column(*contour_and_line_plots(dim_widget.value, val_widget.value))

            def update_val(event):
                column[2] = pn.Column(*contour_and_line_plots(dim_widget.value, val_widget.value))

            dim_widget.param.watch(update_dim, 'value')
            val_widget.param.watch(update_val, 'value')

            return column

        pos_plots = interactive_plots(xr_pos)
        freq_plots = interactive_plots(xr_freq)

        return pn.Column(
            pn.pane.DataFrame(value_specs, index = False, width = 800),
            pn.Row(
                pos_plots, freq_plots
            )
        )
