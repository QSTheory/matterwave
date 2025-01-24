import numpy as np
from typing import get_args
import pytest

import fftarray as fa
from matterwave import scalar_product, norm

@pytest.mark.parametrize("space", get_args(fa.Space))
def test_scalar_product_norm(space: fa.Space) -> None:
    dim = fa.dim("x", n = 4, d_pos = 1., pos_min = 0.3, freq_min = 0.7)
    arr = fa.coords_from_dim(dim, space)
    arr_conj = fa.conj(arr)
    sn = scalar_product(arr, arr_conj)
    n = norm(arr)
    assert sn == n

@pytest.mark.parametrize("spaceA", get_args(fa.Space))
@pytest.mark.parametrize("spaceB", get_args(fa.Space))
def test_scalar_product_orth(spaceA: fa.Space, spaceB: fa.Space) -> None:
    x_dim = fa.dim_from_constraints("x", n=5, pos_middle=np.pi, pos_extent=2*np.pi, freq_middle=np.pi)
    y_dim = fa.dim_from_constraints("y", n=5, pos_middle=np.pi, pos_extent=2*np.pi, freq_middle=np.pi)
    x = fa.coords_from_dim(x_dim, spaceA)
    y = fa.coords_from_dim(y_dim, spaceB)
    arr1 = fa.sin(x) * fa.sin(y)
    arr2 = fa.cos(x) * fa.cos(y)
    sp = scalar_product(arr1, arr2)
    assert np.abs(sp) < 1.e-18

