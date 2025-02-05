from typing import Any
import pytest

import fftarray as fa
from matterwave import split_step
from tests.helpers import XPS, PrecisionSpec, precisions

def test_eager() -> None:
    dim = fa.dim("x", n = 4, d_pos = 1., pos_min = 0.3, freq_min = 0.7)
    arr = fa.coords_from_dim(dim, "pos")

    # TODO: Needs lazy implemented to work
    # assert split_step(arr, dt=1., mass=1., V=arr)._factors_applied == (False,)
    psi = arr.into_eager(True)
    V = arr.into_eager(True)
    assert split_step(psi, dt=1., mass=1., V=V)._factors_applied == (True,)

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("precision", precisions)
def test_psi(xp: Any, precision: PrecisionSpec) -> None:
    dim = fa.dim("x", n = 4, d_pos = 1., pos_min = 0.3, freq_min = 0.7)
    arr = fa.coords_from_dim(dim, "pos", xp=xp, dtype=getattr(xp, precision)).into_eager(False)
    # TODO Actually test the result and not just that it does not crash.
    def pot_func(psi: fa.Array) -> fa.Array:
        return fa.abs(psi)**2
    split_step(arr, dt=1., mass=1., V=pot_func)
