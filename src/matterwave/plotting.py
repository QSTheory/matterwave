
try:
    from ._src.fftarray_plotting import (
        plot_array as plot_array,
        generate_panel_plot as generate_panel_plot,
    )
except ModuleNotFoundError as e:
      raise ModuleNotFoundError("You need to install `matterwave[plotting]` to use the plotting routines.") from e

__all__ = [
   g for g in globals() if not g.startswith("_")
]
