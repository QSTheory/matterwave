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
from .split_step import split_step, propagate
from .wf_tools import norm, normalize,  get_ground_state_ho, get_e_kin, scalar_product, expectation_value