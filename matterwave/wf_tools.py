from fftarray import FFTArray, FFTDimension, Space
# from fftarray.tools import shift_frequency
from scipy.constants import pi, hbar, Boltzmann
import numpy as np
from typing import Optional, Callable, Tuple, List
from functools import reduce

def norm(wf: FFTArray) -> float:
    """Compute the norm of the given FFTWave in its current space.

    Parameters
    ----------
    wf : FFTWave
        The FFTWave.

    Returns
    -------
    float
        The norm in position space.

    See Also
    --------
    matterwave.wf_tools.normalize
    """
    abs_sq: FFTArray = np.abs(wf)**2 # type: ignore
    return integrate(abs_sq)

def integrate(abs_sq: FFTArray) -> float:
    """Integrate the given |wf|^2 in the space of wf.

    Parameters
    ----------
    wf : FFTWave
        The FFTWave.

    Returns
    -------
    float
        The integral of the given |wf|^2 in the space of wf

    """
    assert abs_sq.values.dtype == abs_sq.tlib.real_type
    reduced = abs_sq.tlib.numpy_ufuncs.sum(abs_sq.values)

    if _scalar_space(abs_sq) == "pos":
        return reduced * abs_sq.d_pos
    else:
        return reduced * abs_sq.d_freq

def normalize(wf: FFTArray) -> FFTArray:
    """Normalize the FFTWave.

    Parameters
    ----------
    wf : FFTWave
        The initial FFTWave.

    Returns
    -------
    FFTWave
        The normalized FFTWave.

    See Also
    --------
    matterwave.wf_tools.norm_pos_space
    matterwave.wf_tools.norm_freq_space
    """
    norm_factor = wf.tlib.numpy_ufuncs.sqrt(1./norm(wf))
    return wf * norm_factor

def get_e_kin(wf: FFTArray, m: float, return_microK: bool = False) -> float:
    """Compute the kinetic energy of the given FFTWave with the given mass.

    Parameters
    ----------
    wf : FFTWave
        The FFTWave.
    m : float
        The mass of the FFTWave.
    return_microK : bool, optional
        Return the kinetic energy in microK instead of Joule.
        This option exists since returning it in Joule does not work in single
        precision (fp32) due to internal accuracy limitations, by default False.

    Returns
    -------
    float
        The kinetic energy.

    See Also
    --------
    fftarray.fft_wave.FFTWave.expectation_value_freq
    """
    # Move hbar**2/(2*m) until after accumulation to allow accumulation also in fp32.
    # Otherwise the individual values typically underflow to zero.
    kin_op = reduce(lambda a,b: a+b, [(2*np.pi*dim.fft_array(space="freq"))**2. for dim in wf.dims])
    post_factor = hbar**2/(2*m)
    if return_microK:
        post_factor /= (Boltzmann * 1e-6)
    return expectation_value(wf, kin_op) * post_factor

def get_ground_state(dim: FFTDimension, *,
                    omega: Optional[float] = None,
                    sigma_p: Optional[float] = None,
                    mass: float,
                ) -> FFTArray:
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
    fftarray.fft_wave.FFTWave.map_pos_space
    """
    if omega and sigma_p:
        raise ValueError("You can only specify ground state width either as omega or sigma_p, not both.")
    if sigma_p:
        omega =  2 * (sigma_p**2) / (mass * hbar)
    assert omega, "Momentum width has not been specified via either sigma_p or omega."

    wf = (mass * omega / (pi*hbar))**(1./4.) * np.exp(-(mass * omega * (dim.fft_array(space="pos")**2.)/(2.*hbar))+0.j)

    wf = normalize(wf)
    return wf


def scalar_product(a: FFTArray, b: FFTArray) -> float:
    assert a.space == b.space
    bra_ket: FFTArray = np.conj(a)*b # type: ignore
    reduced = bra_ket.tlib.numpy_ufuncs.real(bra_ket.tlib.numpy_ufuncs.sum(bra_ket.values))

    if _scalar_space(a) == "pos":
        return reduced * bra_ket.d_pos
    else:
        return reduced * bra_ket.d_freq

def _scalar_space(wf: FFTArray) -> Space:
    if all([dim_space == "pos" for dim_space in wf.space]):
        return "pos"
    elif all([dim_space == "freq" for dim_space in wf.space]):
        return "freq"
    raise ValueError(f"Wave function must have same space in all dimensions.")

def expectation_value(wf: FFTArray, op: FFTArray) -> float:
    """
        Compute the expectation value of the given diagonal operator on the FFTWave in the space of the operator.

        Parameters
        ----------
        wf : FFTWave
            The FFTWave.
        op : FFTWave
            The diagonal operator.

        Returns
        -------
        float
            The expectation value of the given diagonal operator.
    """


    if _scalar_space(op) == "pos":
        wf_in_op_space: FFTArray = wf.into(space="pos")
    else:
        wf_in_op_space = wf.into(space="freq")

    # We can move the operator out of the scalar product because it is diagonal.
    # This way we can use the more efficient computation of wf_abs_sq.
    wf_abs_sq: FFTArray = np.abs(wf_in_op_space)**2 # type: ignore
    return integrate(wf_abs_sq*op)
