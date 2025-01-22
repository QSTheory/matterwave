"""
.. rubric:: The objects inside the tables can be imported directly from :py:mod:`matterwave`:

Provided by :py:mod:`matterwave.split_step`:

.. currentmodule:: matterwave.split_step

.. autosummary::
   :nosignatures:

   split_step
   propagate


Provided by :py:mod:`matterwave.wf_tools`:

.. currentmodule:: matterwave.wf_tools

.. autosummary::
   :nosignatures:

   norm_pos_space
   norm_freq_space
   normalize
   get_e_kin
   set_ground_state

Example:

.. code-block:: python

	>>> from matterwave import split_step
	>>> from matterwave import normalize

"""
from .split_step import (
   propagate as propagate,
   split_step as split_step,
)
from .wf_tools import (
   expectation_value as expectation_value,
   get_e_kin as get_e_kin,
   get_ground_state_ho as get_ground_state_ho,
   norm as norm,
   normalize as normalize,
   scalar_product as scalar_product,
)

__all__ = [
    g for g in globals() if not g.startswith("_")
]
