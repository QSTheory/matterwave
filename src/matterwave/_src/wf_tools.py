import fftarray as fa
from scipy.constants import pi, hbar, Boltzmann
import numpy as np
from typing import Optional, Any
from functools import reduce

def norm(psi: fa.Array) -> float:
    """Compute the norm of the given Array in its current space.

    Parameters
    ----------
    psi : Array
        The wave function.

    Returns
    -------
    float
        The norm of the Array.

    See Also
    --------
    matterwave.wf_tools.normalize
    """
    abs_sq: fa.Array = fa.abs(psi)**2 # type: ignore
    arr_norm: fa.Array = fa.integrate(abs_sq)
    return arr_norm.values(())

def normalize(psi: fa.Array) -> fa.Array:
    """Normalize the wave function.

    Parameters
    ----------
    psi : Array
        The initial wave function.

    Returns
    -------
    Array
        The normalized wave function.

    See Also
    --------
    matterwave.wf_tools.norm
    """
    norm_factor = psi.xp.sqrt(1./norm(psi))
    return psi * norm_factor

def get_e_kin(psi: fa.Array, m: float, return_microK: bool = False) -> fa.Array:
    """Compute the kinetic energy of the given FFTWave with the given mass.

    Parameters
    ----------
    psi : Array
        The wave function.
    m : float
        The mass of the wave function.
    return_microK : bool, optional
        Return the kinetic energy in microK instead of Joule.
        This option exists since returning it in Joule does not work in single
        precision (fp32) due to internal accuracy limitations, by default False.

    Returns
    -------
    Array
        The kinetic energy.

    See Also
    --------
    matterwave.expectation_value
    """
    # Move hbar**2/(2*m) until after accumulation to allow accumulation also in fp32.
    # Otherwise the individual values typically underflow to zero.
    kin_op = reduce(lambda a,b: a+b, [(2*np.pi*fa.coords_from_arr(psi, dim.name, "freq"))**2. for dim in psi.dims])
    post_factor = hbar**2/(2*m)
    if return_microK:
        post_factor /= (Boltzmann * 1e-6)
    return expectation_value(psi, kin_op) * post_factor

def get_ground_state_ho(
            dim: fa.Dimension,
            *,
            xp: Optional[Any] = None,
            dtype: Optional[Any] = None,
            omega: Optional[float] = None,
            sigma_p: Optional[float] = None,
            mass: float,
        ) -> fa.Array:
    """Sets the wavefunction to the ground state of the isotropic n-dimensional
    quantum harmonic oscillator (QHO). n equals the dimension of the given
    FFTWave. Either ``omega`` or ``sigma_p`` has to be specified.
    The ground state is centered at the origin in posiion and frequency space.

    .. math::

        \Psi (\\vec{r}) = \\left( \\frac{m \omega}{\pi \hbar}  \\right)^\\frac{1}{4} e^{-\\frac{m\omega \\vec{r}^2}{2\hbar}}

    Parameters
    ----------
    wf : FFTWave
        The initial FFTWave.
    mass : float
        The mass of the FFTWave.
    omega : Optional[float], optional
        The angular frequency of the QHO, by default None
    sigma_p : Optional[float], optional
        The momentum uncertainty, by default None

    Returns
    -------
    FFTWave
        The ground state FFTWave.

    Raises
    ------
    ValueError
        If ``omega`` and ``sigma_p`` are both specified.

    See Also
    --------
    fftarray.coords_from_dim
    """
    if omega and sigma_p:
        raise ValueError("You can only specify ground state width either as omega or sigma_p, not both.")
    if sigma_p:
        omega =  2 * (sigma_p**2) / (mass * hbar)
    assert omega, "Momentum width has not been specified via either sigma_p or omega."
    x: fa.Array = fa.coords_from_dim(dim, "pos", xp=xp, dtype=dtype)
    psi: fa.Array = (mass * omega / (pi*hbar))**(1./4.) * fa.exp(-(mass * omega * (x**2.)/(2.*hbar)))
    psi = normalize(psi)
    return psi


def scalar_product(a: fa.Array, b: fa.Array) -> float:
    """Take the scalar product between two wave functions.

    Parameters
    ----------
    a : fa.Array
        Wavefunction <pos|a>
    b : fa.Array
        Wavefunction <pos|b>

    Returns
    -------
    float
        Scalar product.
    """
    assert a.spaces == b.spaces
    bra_ket: fa.Array = fa.conj(a)*b # type: ignore
    return fa.real(fa.integrate(bra_ket)).values(())


def expectation_value(psi: fa.Array, op: fa.Array) -> float:
    """Compute the expectation value of the given diagonal operator on the
    fa.Array in the space of the operator.

    Parameters
    ----------
    wf : fa.Array
        The wave function.
    op : fa.Array
        The diagonal operator.

    Returns
    -------
    float
        The expectation value of the given diagonal operator.
    """
    psi_in_op_space = psi.into_space(op.spaces)
    # We can move the operator out of the scalar product because it is diagonal.
    # This way we can use the more efficient computation of psi_abs_sq.
    psi_abs_sq: fa.Array = fa.abs(psi_in_op_space)**2 # type: ignore
    return fa.integrate(psi_abs_sq*op).values(())
