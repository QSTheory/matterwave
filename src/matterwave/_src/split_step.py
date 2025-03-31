import numpy as np
from scipy.constants import hbar
from typing import Union

from .wf_tools import normalize
import fftarray as fa
from functools import reduce

# Get the position propagator for a specified kernel and dt.
def get_V_prop(V: fa.Array, dt: complex) -> fa.Array:
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
    return fa.exp((-1.j / hbar * dt) * V.into_dtype("complex"))


# TODO Performance optimization of split-step merging
def split_step(
        psi: fa.Array,
        *,
        dt: Union[float, complex],
        mass: float,
        V: fa.Array,
    ) -> fa.Array:
    """Split-step is a pseudo-spectral method to solve the time-dependent
    Schrödinger equation. The time evolution of a wavefunction is given by:

    .. math::

        \Psi (x,t+dt) = e^{-\\frac{i}{\hbar}H dt} \Psi (x,t).

    The time evolution operator can be approximated by [1]_

    .. math::

        e^{-\\frac{i}{\hbar}H dt} = e^{-\\frac{i}{2\hbar}\hat V dt} e^{-\\frac{i}{\hbar}\hat T dt} e^{-\\frac{i}{2\hbar}\hat V dt} + \mathcal O (dt^3).

    Note that the kinetic energy operator :math:`\hat T` is diagonal in
    frequency space and that :math:`\hat V` is diagonal in position space. The
    split-step method utilizes this as follows.

    1. Apply :math:`e^{-\\frac{i}{2\hbar}\hat V dt}` in position space.
    2. Perform an FFT to get the wavefunction in frequency space.
    3. Apply :math:`e^{-\\frac{i}{\hbar}\hat T dt}` in frequency space.
    4. Perform an inverse FFT to get the wavefunction in position space.
    5. Apply :math:`e^{-\\frac{i}{2\hbar}\hat V dt}` in position space.

    By this, the computation of the wavefunction's time evolution is
    significantly faster. Note that the timestep :math:`dt` should be chosen
    small to reduce the overall error.

    Parameters
    ----------
    psi : fa.Array
        The initial wavefunction :math:`\Psi(x,t)`.
    dt : float
        The timestep :math:`dt`.
    mass : float
        The wavefunction's mass.
    V : fa.Array
        The potential in position space.

    Returns
    -------
    fa.Array
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
    This example shows how to perform a split-step application. For more
    extensive explanations to this example, please visit
    `Examples <https://gitlab.projekt.uni-hannover.de/iqo-seckmeyer/unified_wf_2/-/tree/master/examples>`_.

    >>> from matterwave import split_step, set_ground_state
    >>> import fftarray as fa
    >>> from matterwave.rb87 import m as mass_rb87
    >>> from scipy.constants import pi
    >>> # Initialize constants
    >>> mass = mass_rb87 # kg
    >>> omega_x_init = 2.*pi*0.1 # Hz
    >>> omega_x = 2.*pi # Hz
    >>> dt = 1e-4 # s
    >>> # Define the Dimension
    >>> dim = fa.dim_from_constraints("x", pos_min = -200e-6, pos_max = 200e-6, freq_offset = 0., n = 2048)
    >>> # Get the coordinates as an Array
    >>> x = fa.coords_from_dim(dim, "pos")
    >>> # Define the potential
    >>> V = 0.5 * mass * omega_x**2. * x**2.
    >>> # Initialize the wavefunction
    >>> psi_init = get_ground_state_ho(dim, omega=omega_x_init, mass=mass)
    >>> # Perform the split-step
    >>> psi_final = split_step(psi_init, dt=dt, mass=mass, V=V)
    """
    # Compute potential propagator from potential with half the time step
    V_prop = get_V_prop(V = V, dt = 0.5*dt)

    # Apply half potential propagator
    psi = psi.into_space("pos")
    psi = psi * V_prop

    # Apply kinetic propagator
    psi = propagate(psi, dt = dt, mass = mass)

    # Apply half potential propagator
    psi = psi.into_space("pos")
    psi = psi * V_prop

    return psi

def split_step_imag_time(
        psi: fa.Array,
        *,
        dt: float,
        mass: float,
        V: fa.Array,
    ) -> fa.Array:
    """Imaginary time evolution: :math:`dt \\rightarrow -i dt` using split-step.
    Normalization is included.

    Parameters
    ----------
    psi : fa.Array
        The initial wavefunction :math:`\Psi(x,t)`.
    dt : float
        The timestep :math:`dt`.
    mass : float
        The wavefunction's mass.
    V : fa.Array
        The potential in position space.

    Returns
    -------
    fa.Array
        The wavefunction after one imaginary time evolution step.

    See Also
    --------
    matterwave.split_step.split_step :
        Used to propagate the wavefunction.
    matterwave.wf_tools.normalize :
        Used to normalize the wavefunction.

    Example
    -------
    This example shows how to perform a split-step application with imaginary
    time. For more extensive explanations to this example, please visit
    `Examples <https://gitlab.projekt.uni-hannover.de/iqo-seckmeyer/unified_wf_2/-/tree/master/examples>`_.

    >>> from matterwave import split_step, set_ground_state
    >>> import fftarray as fa
    >>> from matterwave.rb87 import m as mass_rb87
    >>> from scipy.constants import pi
    >>> # Initialize constants
    >>> mass = mass_rb87 # kg
    >>> omega_x_init = 2.*pi*0.1 # Hz
    >>> omega_x = 2.*pi # Hz
    >>> dt = 1e-4 # s
    >>> # Define the Dimension
    >>> dim = fa.dim_from_constraints("x", pos_min = -200e-6, pos_max = 200e-6, freq_offset = 0., n = 2048)
    >>> # Get the coordinates as an Array
    >>> x = fa.coords_from_dim(dim, "pos")
    >>> # Define the potential
    >>> V = 0.5 * mass * omega_x**2. * x**2.
    >>> # Initialize the wavefunction
    >>> psi_init = get_ground_state_ho(dim, omega=omega_x_init, mass=mass)
    >>> # Perform the split-step
    >>> psi_final = split_step_imag_time(psi_init, dt=dt, mass=mass, V=V)
    """
    psi = split_step(
        psi,
        dt = -1.j * dt, # must be with minus since i**2 == -1
        mass = mass,
        V = V,
    )
    psi = normalize(psi)
    return psi


# TODO: Do a proper numerical analysis.
# Maybe use https://herbie.uwplse.org/
# TODO Benchmark performance
def propagate(psi: fa.Array, *, dt: Union[float, complex], mass: float) -> fa.Array:
    """Freely propagates the given wavefunction in time:

    .. math::

        \\Psi (x,t+dt) = e^{-\\frac{i}{\hbar}\hat T dt} \Psi (x,t).

    Parameters
    ----------
    psi : fa.Array
        The initial wavefunction :math:`\Psi(x,t)`.
    dt : Union[float, complex]
        The timestep :math:`dt`.
    mass : float
        The wavefunction's mass.

    Returns
    -------
    fa.Array
        The freely propagated wavefunction :math:`\Psi(x,t+dt)`.
    """
    # p_sq = k_sq * hbar^2
    # Propagator in p: value * jnp.exp(-1.j * dt * p_sq / (2*mass*hbar))
    # => This formulation uses less numerical range to enable single precision floats
    # In 3D: kx**2+ky**2+kz**2
    k_sq = reduce(lambda a,b: a+b, [(2*np.pi*fa.coords_from_arr(psi, dim.name, "freq"))**2. for dim in psi.dims])
    return psi.into_space("freq") * fa.exp((-1.j * dt * hbar / (2*mass)) * k_sq.into_dtype("complex")) # type: ignore

