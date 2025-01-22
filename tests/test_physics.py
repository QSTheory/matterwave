import fftarray as fa
from matterwave.split_step import split_step
from matterwave.wf_tools import expectation_value, get_ground_state_ho, get_e_kin, norm
from matterwave.rb87 import m as m_rb87
from matterwave.tests.helpers import XPS, PrecisionSpec, precisions

import numpy as np
from scipy.constants import hbar, pi, Boltzmann
import pytest
from typing import Any
from array_api_compat import is_jax_array

# Check whether a 1d FFTWave initialization in x with mapping
# the 1d first excited state of the harmonic oscillator correctly
# implements the split_step method by looking at the wavefunction's
# total energy after a few steps

@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("eager", [False, True])
def test_1d_x_split_step(xp: Any, precision: PrecisionSpec, eager: bool) -> None:

    mass = m_rb87
    omega_x = 2*pi

    x_dim = fa.dim_from_constraints("x",
        pos_min=-100e-6,
        pos_max=100e-6,
        freq_middle=0.,
        n=1024,
        dynamically_traced_coords=False
    )
    x = fa.coords_from_dim(x_dim, "pos", xp=xp, dtype=getattr(xp, precision)).into_eager(eager)
    psi = 1./np.sqrt(2.)*(mass*omega_x/(pi*hbar))**(1./4.) * \
            fa.exp(-mass*omega_x*x**2./(2.*hbar)) * \
                2*np.sqrt(mass*omega_x/hbar)*x

    harmonic_potential_1d = 0.5 * mass * omega_x**2. * x**2.

    def split_step_scan_iteration(psi, *_):
        psi = split_step(psi, mass=mass, dt=1e-5, V=harmonic_potential_1d)
        return psi, None

    def nojax_scan(psi: fa.Array):
        for _ in range(100):
            psi, _ = split_step_scan_iteration(psi)
        return psi, None

    if is_jax_array(psi._values):
        from jax.lax import scan
        psi, _ = scan(
            f=split_step_scan_iteration,
            init=psi.into_space("freq").into_factors_applied(eager),
            xs=None,
            length=100,
        )
    else:
        psi, _ = nojax_scan(psi)

    e_pot = expectation_value(psi, harmonic_potential_1d)
    e_kin = get_e_kin(psi, m=mass)

    assert (e_pot + e_kin).values("pos") == pytest.approx(hbar*omega_x*3./2.)


# # Test the split step method for imaginary time steps. Start with a ground state
# # of different angular frequency than the system and evolve it towards the
# # system's ground state. The resulting total energy should be lower than the
# # initial one. Additionally, it is checked whether the resulting wavefunction is
# # normalized.
@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("eager", [False, True])
def test_1d_split_step_complex(xp: Any, precision: PrecisionSpec, eager: bool) -> None:

    mass = m_rb87
    omega_x_init = 2.*pi # angular freq. for initial (ground) state
    omega_x = 2.*pi*0.1 # angular freq. for desired ground state
    x_dim = fa.dim_from_constraints("x",
        pos_min=-200e-6,
        pos_max=200e-6,
        freq_middle=0.,
        n=2048,
        dynamically_traced_coords=False
    )

    psi = get_ground_state_ho(
        dim=x_dim,
        xp=xp,
        dtype=getattr(xp, precision),
        omega=omega_x_init,
        mass=mass,
    ).into_eager(eager)
    x = fa.coords_from_arr(psi, x_dim.name, "pos")

    V = 0.5 * mass * omega_x**2. * x**2.
    def total_energy(psi: fa.Array) -> float:
        E_kin = get_e_kin(psi, m=mass, return_microK=True).values("pos")
        E_pot = expectation_value(psi, V).values("pos") / (Boltzmann * 1e-6)
        return E_kin + E_pot

    energy_before = total_energy(psi)

    def step(psi: fa.Array, *_):
        psi = split_step(psi, dt=1e-4, mass=mass, V=V, is_complex=True)
        return psi, None

    def nojax_scan(psi: fa.Array) -> fa.Array:
        for _ in range(128):
            psi, _ = step(psi)
        return psi, None

    if is_jax_array(psi._values):
        from jax.lax import scan
        psi, _ = scan(
            f=step,
            init=psi.into_space("freq").into_factors_applied(eager),
            xs=None,
            length=128,
        )
    else:
        psi, _ = nojax_scan(psi)

    energy_after = total_energy(psi)
    # check whether wafefunction is normalized
    np.testing.assert_array_almost_equal(float(norm(psi).values("pos")), 1.)
    # check if energy is reduced (iteration towards ground state successfull)
    assert energy_after < energy_before

# # Test the set_ground_state method. Initializes a ground state with angular
# # frequency 2*pi. Then, the total energy of the returned state is computed to
# # compare it to the analytical solution. Also it is checked whether the returned
# # wavefunction is normalized.
@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("eager", [False, True])
def test_1d_set_ground_state(xp: Any, precision: PrecisionSpec, eager: bool) -> None:

    mass = m_rb87
    omega_x = 2*pi
    x_dim = fa.dim_from_constraints("x",
        pos_min=-200e-6,
        pos_max=200e-6,
        freq_middle=0.,
        n=2048,
        dynamically_traced_coords=False
    )

    psi = get_ground_state_ho(
        dim=x_dim,
        xp=xp,
        dtype=getattr(xp, precision),
        omega=omega_x,
        mass=mass,
    ).into_eager(eager)
    x = fa.coords_from_arr(psi, x_dim.name, "pos")

    # quantum harmonic oscillator
    V = 0.5 * mass * omega_x**2. * x**2.
    # check if ground state is normalized
    np.testing.assert_array_almost_equal(float(norm(psi).values("pos")), 1)
    E_kin = get_e_kin(psi, m=mass, return_microK=True)
    E_pot = expectation_value(psi, V) / (Boltzmann * 1e-6)
    E_tot = (E_kin + E_pot).values("pos")
    E_tot_analytical = 0.5*omega_x*hbar / (Boltzmann * 1e-6)
    # check if its energy is equal to the analytical solution
    np.testing.assert_array_almost_equal(float(E_tot), float(E_tot_analytical))
