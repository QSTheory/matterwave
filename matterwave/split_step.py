import numpy as np
from scipy.constants import hbar, h
from typing import Callable, Union, Any

from .wf_tools import normalize
from fftarray import FFTArray, PosArray, FreqArray
from fftarray.fft_array import PhaseFactors
from functools import reduce

# Get the position propagator for a specified kernel and dt.
def get_V_prop(V: FFTArray, dt: complex) -> FFTArray:
    """The propagator for the potential: :math:`e^{-\\frac{i}{\hbar} \hat V dt}`
    where the potential energy operator :math:`\hat V` is defined by `V_kernel`.

    :meta private:

    Parameters
    ----------
    V_kernel : Callable[..., complex]
        Returns the potential energy propagator.

    Returns
    -------
    Callable[..., complex]
        The function defining the potential energy propagator. Takes `dt` as
        required argument.
    """
    return np.exp((-1.j / hbar * dt) * V)


# TODO Performance optimization of split-step merging
def split_step(wf: FFTArray, *,
               dt: float,
               mass: float,
               V: Union[PosArray, Callable[[PosArray], Any]], # numpy ufuncs distort the type signature, actually this should be PosArray
               is_complex: bool = False) -> FreqArray:
    """Split-step is a pseudo-spectral method to solve the time-dependent
    Schrödinger equation. The time evolution of a wavefunction is given by:

    .. math::

        \Psi (x,t+dt) = e^{-\\frac{i}{\hbar}H dt} \Psi (x,t).

    The time evolution operator can be approximated by [1]_

    .. math::

        e^{-\\frac{i}{\hbar}H dt} = e^{-\\frac{i}{2\hbar}\hat T dt} e^{-\\frac{i}{\hbar}\hat V dt} e^{-\\frac{i}{2\hbar}\hat T dt} + \mathcal O (dt^3).

    Note that the kinetic energy operator :math:`\hat T` is diagonal in
    momentum space and that :math:`\hat V` is diagonal in position space. The
    split-step method utilizes this as follows.

    1. Apply :math:`e^{-\\frac{i}{2\hbar}\hat T dt}` in momentum space.
    2. Perform an FFT to get the wavefunction in position space.
    3. Apply :math:`e^{-\\frac{i}{\hbar}\hat V dt}` in position space.
    4. Perform an inverse FFT to get the wavefunction in momentum space.
    5. Apply :math:`e^{-\\frac{i}{2\hbar}\hat T dt}` in momentum space.

    By this, the computation of the wavefunction's time evolution is
    significantly faster. Note that the timestep :math:`dt` should be chosen
    small to reduce the overall error.

    Parameters
    ----------
    wf : FFTWave
        The initial wavefunction :math:`\Psi(x,t)`.
    dt : float
        The timestep :math:`dt`.
    mass : float
        The wavefunction's mass.
    V_kernel : Callable[..., complex]
        The kernel defining the potential V.
    kwargs : Dict[str, Any]
        Additional arguments to pass to `V_kernel`.
    is_complex : bool, optional
        Imaginary time evolution: :math:`dt \\rightarrow -i dt`
        (the normalization of the wavefunction is included), by default False

    Returns
    -------
    FFTWave
        The wavefunction evolved in time: :math:`\Psi(x,t+dt)`.

    See Also
    --------
    matterwave.split_step.propagate :
        Used to freely propagate the wavefunction.
    jax.lax.scan
        Useful function to speed up an iteration.

    References
    ----------
    .. [1] M.D Feit, J.A Fleck, A Steiger, "Solution of the Schrödinger equation
       by a spectral method", Journal of Computational Physics, Volume 47, Issue
       3, 1982, Pages 412-433, ISSN 0021-9991,
       https://doi.org/10.1016/0021-9991(82)90091-2.

    Example
    -------
    This example shows how to perform a split-step application with imaginary
    time. For more extensive explanations to this example, please visit
    `Examples <https://gitlab.projekt.uni-hannover.de/iqo-seckmeyer/unified_wf_2/-/tree/master/examples>`_.

    >>> from matterwave import split_step, set_ground_state
    >>> from fftarray import FFTWave
    >>> from matterwave.rb87 import m as mass_rb87
    >>> from scipy.constants import pi
    >>> # Initialize constants
    >>> mass = mass_rb87 # kg
    >>> omega_x_init = 2.*pi*0.1 # Hz
    >>> omega_x = 2.*pi # Hz
    >>> dt = 1e-4 # s
    >>> # Define the potential
    >>> def V_kernel(value: complex, x: float):
    >>>     return 0.5 * mass * omega_x**2. * x**2.
    >>> # Initialize the wavefunction
    >>> wf_init = FFTWave(x_min = -200e-6, x_max = 200e-6, kx_offset = 0., nx = 2048)
    >>> wf_init = set_ground_state(wf_init, omega=omega_x_init, mass=mass)
    >>> # Perform the split-step
    >>> wf_final = split_step(wf_init, dt=dt, mass=mass, V_kernel=V_kernel, is_complex=True)
    """

    if is_complex:
        cmplx_factor = -1.j # Must be with minus since i**2 == -1
    else:
        cmplx_factor = 1.

    # Apply half kinetic propagator
    wf = propagate(wf, dt = cmplx_factor * 0.5*dt, mass = mass)

    # Apply potential propagator
    # TODO
    wf = wf.pos_array()
    if isinstance(V, PosArray):
        V_prop = get_V_prop(V = V, dt = cmplx_factor * dt)
    else:
        V_prop = get_V_prop(V = V(wf), dt = cmplx_factor * dt)
    wf = wf * V_prop

    # Apply half kinetic propagator
    wf = propagate(wf, dt = cmplx_factor * 0.5*dt, mass = mass)

    if is_complex:
        wf = normalize(wf)
    assert isinstance(wf, FreqArray)
    return wf



# def mom_propagator(dt: float, mass: float):


# return p_kernel

# TODO: Do a proper numerical analysis.
# Maybe use https://herbie.uwplse.org/
# TODO Benchmark performance
def propagate(wf: FFTArray, *, dt: Union[float, complex], mass: float) -> FreqArray:
    """Freely propagates the given wavefunction in time:

    .. math::

        \\Psi (x,t+dt) = e^{-\\frac{i}{\hbar}\hat T dt} \Psi (x,t).

    Parameters
    ----------
    wf : FFTWave
        The initial wavefunction :math:`\Psi(x,t)`.
    dt : Union[float, complex]
        The timestep :math:`dt`.
    mass : float
        The wavefunction's mass.

    Returns
    -------
    FFTWave
        The freely propagated wavefunction :math:`\Psi(x,t+dt)`.
    """
    # TODO: Use lazy phase factor
    # p_sq = k_sq * hbar^2
    # Propagator in p: value * jnp.exp(-1.j * dt * p_sq / (2*mass*hbar))
    # => This formulation uses less numerical range to enable single precision floats

    # In 3D: kx**2+ky**2+kz**2
    k_sq = reduce(lambda a,b: a+b, [(2*np.pi*dim.freq_array())**2. for dim in wf.dims])
    return wf.freq_array() * np.exp((-1.j * dt * hbar / (2*mass)) * k_sq) # type: ignore


