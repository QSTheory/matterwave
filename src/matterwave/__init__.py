"""
.. rubric:: The objects inside the tables can be imported directly from :py:mod:`matterwave`:

.. autosummary::
   :nosignatures:

   expectation_value
   get_e_kin
   get_ground_state_ho
   norm
   normalize
   propagate
   split_step

Example:

.. code-block:: python

	>>> from matterwave import split_step
	>>> from matterwave import normalize

"""
from ._src.split_step import (
   propagate as propagate,
   split_step as split_step,
)

from ._src.wf_tools import (
   expectation_value as expectation_value,
   get_e_kin as get_e_kin,
   get_ground_state_ho as get_ground_state_ho,
   norm as norm,
   normalize as normalize,
   scalar_product as scalar_product,
)

from ._src.fftarray_plotting import (
   plot_array as plot_array,
   generate_panel_plot as generate_panel_plot,
)

__all__ = [
   g for g in globals() if not g.startswith("_")
]
