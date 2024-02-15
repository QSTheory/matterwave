from jax.config import config
config.update("jax_enable_x64", True)

from jax.lax import scan

from fftarray.fft_constraint_solver import fft_dim_from_constraints
from fftarray.backends.jax_backend import JaxTensorLib
from fftarray.backends.np_backend import NumpyTensorLib
from fftarray.backends.pyfftw_backend import PyFFTWTensorLib
from matterwave import split_step, get_e_kin, norm
from matterwave.wf_tools import expectation_value, get_ground_state_ho

# Math
import numpy as np

# Constants
from scipy.constants import hbar, pi, Boltzmann
from matterwave.rb87 import m as m_rb87

# Testing
import pytest

# Check whether a 1d FFTWave initialization in x with mapping
# the 1d first excited state of the harmonic oscillator correctly
# implements the split_step method by looking at the wavefunction's
# total energy after a few steps

backends = ["numpy", "jax", "pyfftw"]

def get_tensor_lib(backend: str):
    if backend == "numpy":
        return NumpyTensorLib(precision="fp64")
    elif backend == "jax":
        return JaxTensorLib(precision="fp64")
    elif backend == "pyfftw":
        return PyFFTWTensorLib(precision="fp64")

@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("eager", [False, True])
def test_1d_x_split_step(backend: str, eager: bool) -> None:
    tensor_lib = get_tensor_lib(backend)

    mass = m_rb87
    omega_x = 2*pi

    x_dim = fft_dim_from_constraints("x",
        pos_min=-100e-6,
        pos_max=100e-6,
        freq_middle=0.,
        n=1024,
    )
    x = x_dim.fft_array(tlib=tensor_lib, space="pos", eager=eager)
    wf = 1./np.sqrt(2.)*(mass*omega_x/(pi*hbar))**(1./4.) * \
            np.exp(-mass*omega_x*x**2./(2.*hbar)+0.j) * \
                2*np.sqrt(mass*omega_x/hbar)*x

    harmonic_potential_1d = 0.5 * mass * omega_x**2. * x**2.
    def split_step_scan_iteration(wf, *_):
        wf = split_step(wf, mass=mass, dt=1e-5, V=harmonic_potential_1d)
        return wf, None

    if backend == "jax":
        wf, _ = scan(
            f=split_step_scan_iteration,
            init=wf.into(space="freq", factors_applied=False),
            xs=None,
            length=100,
        )
    else:
        for _ in range(100):
            wf, _ = split_step_scan_iteration(wf)

    e_pot = expectation_value(wf, harmonic_potential_1d)
    e_kin = get_e_kin(wf, m=mass)

    assert e_pot + e_kin == pytest.approx(hbar*omega_x*3./2.)

# # Test the split step method for imaginary time steps. Start with a ground state
# # of different angular frequency than the system and evolve it towards the
# # system's ground state. The resulting total energy should be lower than the
# # initial one. Additionally, it is checked whether the resulting wavefunction is
# # normalized.
@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("eager", [False, True])
def test_1d_split_step_complex(backend: str, eager: bool) -> None:
    tensor_lib = get_tensor_lib(backend)

    mass = m_rb87
    omega_x_init = 2.*pi # angular freq. for initial (ground) state
    omega_x = 2.*pi*0.1 # angular freq. for desired ground state
    x_dim = fft_dim_from_constraints("x",
        pos_min=-200e-6,
        pos_max=200e-6,
        freq_middle=0.,
        n=2048,
    )

    wf = get_ground_state_ho(
        dim=x_dim,
        tlib=tensor_lib,
        eager=eager,
        omega=omega_x_init,
        mass=mass,
    )

    x = x_dim.fft_array(tlib=tensor_lib, space="pos", eager=eager)
    V = 0.5 * mass * omega_x**2. * x**2.
    def total_energy(wf):
        E_kin = get_e_kin(wf, m=mass, return_microK=True)
        E_pot = expectation_value(wf, V) / (Boltzmann * 1e-6)
        return E_kin + E_pot
    energy_before = total_energy(wf)
    def step(wf, *_):
        wf = split_step(wf, dt=1e-4, mass=mass, V=V, is_complex=True)
        return wf, None

    if backend == "jax":
        wf, _ = scan(
            f=step,
            # TODO: factors_applied will be False when lazyness is implemented
            init=wf.into(space="freq", factors_applied=False),
            xs=None,
            length=128,
        )
    else:
        for _ in range(128):
            wf, _ = step(wf)

    energy_after = total_energy(wf)
    # check whether wafefunction is normalized
    np.testing.assert_array_almost_equal_nulp(float(norm(wf)), 1., 4)
    # check if energy is reduced (iteration towards ground state successfull)
    assert energy_after < energy_before

# # Test the set_ground_state method. Initializes a ground state with angular
# # frequency 2*pi. Then, the total energy of the returned state is computed to
# # compare it to the analytical solution. Also it is checked whether the returned
# # wavefunction is normalized.
@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("eager", [False, True])
def test_1d_set_ground_state(backend, eager: bool) -> None:
    tensor_lib = get_tensor_lib(backend)

    mass = m_rb87
    omega_x = 2*pi
    x_dim = fft_dim_from_constraints("x",
        pos_min=-200e-6,
        pos_max=200e-6,
        freq_middle=0.,
        n=2048,
    )

    wf = get_ground_state_ho(
        dim=x_dim,
        tlib=tensor_lib,
        eager=eager,
        omega=omega_x,
        mass=mass,
    )
    x = x_dim.fft_array(tlib=tensor_lib, space="pos", eager=eager)
    # quantum harmonic oscillator
    V = 0.5 * mass * omega_x**2. * x**2.
    # check if ground state is normalized
    np.testing.assert_array_almost_equal_nulp(float(norm(wf)), 1, 3)
    E_kin = get_e_kin(wf, m=mass, return_microK=True)
    E_pot = expectation_value(wf, V) / (Boltzmann * 1e-6)
    E_tot = E_kin + E_pot
    E_tot_analytical = 0.5*omega_x*hbar / (Boltzmann * 1e-6)
    # check if its energy is equal to the analytical solution
    np.testing.assert_array_almost_equal_nulp(float(E_tot), float(E_tot_analytical), 13)
