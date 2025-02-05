from typing import Literal, Tuple, get_args

import numpy as np
import array_api_strict
import array_api_compat
import fftarray as fa

XPS = [array_api_strict, array_api_compat.get_namespace(np.asarray(1.))]

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    XPS.append(array_api_compat.get_namespace(jnp.asarray(1.)))
    fa.jax_register_pytree_nodes()
except ImportError:
    pass

DTYPE_NAME = Literal[
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
    "complex64",
    "complex128",
]
dtypes_names_all = get_args(DTYPE_NAME)

PrecisionSpec = Literal["float32", "float64"]
precisions: Tuple[PrecisionSpec] = get_args(PrecisionSpec)
