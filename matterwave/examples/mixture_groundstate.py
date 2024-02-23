# %%
from matterwave.helpers.fftarray_plotting import plot_fftarray
import panel as pn
pn.extension()
from bokeh.io import output_notebook
output_notebook()

from fftarray.fft_constraint_solver import fft_dim_from_constraints
from fftarray import FFTArray, FFTDimension
import numpy as np
from fftarray.backends.jax_backend import JaxTensorLib
from jax import config
from scipy import constants
from matterwave.helpers import generate_panel_plot
from matterwave import get_ground_state_ho

config.update("jax_enable_x64", True)


# %%
hbar: float = constants.hbar
a_0: float = constants.physical_constants['Bohr radius'][0]

# coupling constant (used in GPE)
coupling_fun = lambda m, a: 4 * np.pi * hbar**2 * a / m

# Rubidium 87
m_rb87: float = 86.909 * constants.atomic_mass # The atom's mass in kg.
a_rb87: float = 98 * a_0 # s-wave scattering length
coupling_rb87: float = coupling_fun(m_rb87, a_rb87) # coupling constant (used in GPE)

# Potassium 41
m_k41: float = 40.962 * constants.atomic_mass # The atom's mass in kg.
a_k41: float = 60 * a_0 # s-wave scattering length
coupling_k41: float = coupling_fun(m_k41, a_k41) # coupling constant (used in GPE)

# Interspecies interaction
a_rb87_k41: float = 165.3 * a_0
reduced_mass_rb87_k41 = m_rb87 * m_k41 / (m_rb87 + m_k41)
coupling_rb87_k41: float = 2*np.pi * hbar**2 * a_rb87_k41 / reduced_mass_rb87_k41

# %%
# Define dimensions
x_dim: FFTDimension = fft_dim_from_constraints(
    name="x",
    pos_middle=0,
    freq_middle=0,
    pos_extent=100e-6,
    n=2**12
)
y_dim: FFTDimension = fft_dim_from_constraints(
    name="y",
    pos_middle=0,
    freq_middle=0,
    pos_extent=100e-6,
    n=2**12
)
xarr = x_dim.fft_array(tlib=JaxTensorLib(), space="pos")
generate_panel_plot(xarr)

# %%
trap_frequencies_rb = 2*np.pi*np.array([25, 400])
trap_frequencies_k = np.sqrt(87/41) * trap_frequencies_rb
trap_minimum = np.zeros(2)

tlib = JaxTensorLib()

init_state_rb87_x: FFTArray = get_ground_state_ho(
    x_dim,
    tlib=tlib,
    omega=trap_frequencies_rb[0],
    mass=m_rb87
)

init_state_rb87_y: FFTArray = get_ground_state_ho(
    y_dim,
    tlib=tlib,
    omega=trap_frequencies_rb[1],
    mass=m_rb87
)

init_state_k41_x: FFTArray = get_ground_state_ho(
    x_dim,
    tlib=tlib,
    omega=trap_frequencies_k[0],
    mass=m_k41
)

init_state_k41_y: FFTArray = get_ground_state_ho(
    y_dim,
    tlib=tlib,
    omega=trap_frequencies_k[1],
    mass=m_k41
)

# Combine dimensions into 2-dimensional FFTArrays to represent wavefunctions

init_state_rb87: FFTArray = init_state_rb87_x + init_state_rb87_y
init_state_k41: FFTArray = init_state_k41_x + init_state_k41_y

generate_panel_plot(init_state_rb87)

# %%
generate_panel_plot(init_state_k41)

# %%
# Define imaginary time evolution to find ground state of the system with GPE (first single species)

from functools import partial
from typing import Any, Callable, Tuple
from matterwave import propagate
from matterwave.split_step import get_V_prop
from matterwave.wf_tools import normalize
import jax.numpy as jnp

x_fftarray = x_dim.fft_array(tlib=tlib, space="pos")
y_fftarray = y_dim.fft_array(tlib=tlib, space="pos")

trap_potential_rb87 = 0.5 * m_rb87 * (
    trap_frequencies_rb[0]**2 * (x_fftarray-trap_minimum[0])**2 +
    trap_frequencies_rb[1]**2 * (y_fftarray-trap_minimum[1])**2
)

trap_potential_k41 = 0.5 * m_k41 * (
    trap_frequencies_k[0]**2 * (x_fftarray-trap_minimum[0])**2 +
    trap_frequencies_k[1]**2 * (y_fftarray-trap_minimum[1])**2
)

def gpe_potential(
    pos_state: FFTArray,
    coupling_constant: float,
    trap_potential: FFTArray,
    num_atoms: int = 1e5,
) -> FFTArray:
    self_interaction = num_atoms * coupling_constant * np.abs(pos_state)**2
    return self_interaction + trap_potential

def gpe_potential_two_species(
    pos_state_1: FFTArray,
    pos_state_2: FFTArray,
    coupling_constant_1: float,
    coupling_constant_12: float,
    trap_potential_1: FFTArray,
    num_atoms_1: int,
    num_atoms_2: int,
) -> FFTArray:
    self_interaction = num_atoms_1 * coupling_constant_1 * np.abs(pos_state_1)**2
    interaction_12 = num_atoms_2 * coupling_constant_12 * np.abs(pos_state_2)**2
    return self_interaction + interaction_12 + trap_potential_1

def imaginary_time_evolution_dual_species(
    state_1: FFTArray,
    state_2: FFTArray,
    dt: float,
    mass_1: float,
    mass_2: float,
    V_1: Callable[[FFTArray, FFTArray], Any],
    V_2: Callable[[FFTArray, FFTArray], Any],
) -> Tuple[FFTArray, FFTArray]:

    complex_factor = -1.j

    # Apply half kinetic propagator
    state_1_new = propagate(state_1, dt = -1.j * 0.5*dt, mass = mass_1)
    state_2_new = propagate(state_2, dt = -1.j * 0.5*dt, mass = mass_2)

    # Apply potential propagator
    state_1_new = state_1_new.into(space="pos")
    state_2_new = state_2_new.into(space="pos")

    V_prop_1 = get_V_prop(V = V_1(state_1_new, state_2.into(space="pos")), dt = complex_factor * dt)
    V_prop_2 = get_V_prop(V = V_2(state_2_new, state_1.into(space="pos")), dt = complex_factor * dt)

    state_1_new = state_1_new * V_prop_1
    state_2_new = state_2_new * V_prop_2

    # Apply half kinetic propagator
    state_1_new = propagate(state_1_new, dt = -1.j * 0.5*dt, mass = mass_1)
    state_2_new = propagate(state_2_new, dt = -1.j * 0.5*dt, mass = mass_2)

    state_1_new = normalize(state_1_new)
    state_2_new = normalize(state_2_new)

    return state_1_new, state_2_new

def imaginary_time_evolution_single_species(
    state: FFTArray,
    dt: float,
    mass: float,
    V: Callable[[FFTArray], Any],
) -> FFTArray:

    complex_factor = -1.j

    # Apply half kinetic propagator
    state = propagate(state, dt = -1.j * 0.5*dt, mass = mass)

    # Apply potential propagator
    state = state.into(space="pos")

    V_prop = get_V_prop(V = V(state), dt = complex_factor * dt)

    state = state * V_prop

    # Apply half kinetic propagator
    state = propagate(state, dt = complex_factor * 0.5*dt, mass = mass)

    state = normalize(state)
    return state

reduced_gpe_potential_rb87 = partial(
    gpe_potential,
    coupling_constant=coupling_rb87,
    trap_potential=trap_potential_rb87,
)

num_atoms_rb87 = 1e5
num_atoms_k41 = 1e5

reduced_dual_species_gpe_potential_rb87 = partial(
    gpe_potential_two_species,
    coupling_constant_1=coupling_rb87,
    coupling_constant_12=coupling_rb87_k41,
    trap_potential_1=trap_potential_rb87,
    num_atoms_1=num_atoms_rb87,
    num_atoms_2=num_atoms_k41,
)
reduced_dual_species_gpe_potential_k41 = partial(
    gpe_potential_two_species,
    coupling_constant_1=coupling_k41,
    coupling_constant_12=coupling_rb87_k41,
    trap_potential_1=trap_potential_k41,
    num_atoms_1=num_atoms_k41,
    num_atoms_2=num_atoms_rb87,
)

init_state_rb87 = normalize(init_state_rb87)
init_state_k41 = normalize(init_state_k41)

# ground_state_rb87 = imaginary_time_evolution_single_species(
#     init_state_rb87,
#     dt=1e-5,
#     mass=m_rb87,
#     V=reduced_gpe_potential_rb87,
# )

# generate_panel_plot(ground_state_rb87)

# ground_state_rb87, ground_state_k41 = imaginary_time_evolution_dual_species(
#     init_state_rb87,
#     init_state_k41,
#     dt=1e-6,
#     mass_1=m_rb87,
#     mass_2=m_k41,
#     V_1=reduced_dual_species_gpe_potential_rb87,
#     V_2=reduced_dual_species_gpe_potential_k41,
# )

# %%
# generate_panel_plot(ground_state_rb87)

# # %%
# generate_panel_plot(ground_state_k41)

# %%
from matterwave.wf_tools import expectation_value, get_e_kin
kb: float = constants.Boltzmann


def imaginary_time_step_single_species(
    state: FFTArray,
    *_
):
    E_kin = get_e_kin(state, m=m_rb87, return_microK=True)
    # calculate the potential energy (and convert to µK)
    state=state.into(space="pos")
    E_pot = expectation_value(state, reduced_gpe_potential_rb87(state)) / (kb * 1e-6)
    state=state.into(space="freq")

    # calculate the total energy
    E_tot = E_kin + E_pot
    # split-step application
    state = imaginary_time_evolution_single_species(
        state,
        dt=1e-5,
        mass=m_rb87,
        V=reduced_gpe_potential_rb87,
    )
    return state, {"E_kin": E_kin, "E_pot": E_pot, "E_tot": E_tot}

def imaginary_time_step_dual_species(
    states: Tuple[FFTArray, FFTArray],
    *_
):
    state_rb87, state_k41 = states
    E_kin_rb87 = get_e_kin(state_rb87, m=m_rb87, return_microK=True)
    E_kin_k41 = get_e_kin(state_k41, m=m_k41, return_microK=True)

    # calculate the potential energy (and convert to µK)
    state_rb87 = state_rb87.into(space="pos")
    state_k41 = state_k41.into(space="pos")
    E_pot_rb87 = expectation_value(
        state_rb87,
        reduced_dual_species_gpe_potential_rb87(
            state_rb87, state_k41
        )
    ) / (kb * 1e-6)
    E_pot_k41 = expectation_value(
        state_k41,
        reduced_dual_species_gpe_potential_k41(
            state_k41, state_rb87
        )
    ) / (kb * 1e-6)

    state_rb87 = state_rb87.into(space="freq")
    state_k41 = state_k41.into(space="freq")

    # calculate the total energy
    E_tot = E_kin_rb87 + E_kin_k41 + E_pot_rb87 + E_pot_k41

    # split-step application
    states = imaginary_time_evolution_dual_species(
        state_1 = state_rb87,
        state_2 = state_k41,
        dt=1e-6,
        mass_1=m_rb87,
        mass_2=m_k41,
        V_1=reduced_dual_species_gpe_potential_rb87,
        V_2=reduced_dual_species_gpe_potential_k41,
    )

    return states, {
        "E_kin_rb87": E_kin_rb87,
        "E_kin_k41": E_kin_k41,
        "E_pot_rb87": E_pot_rb87,
        "E_pot_k41": E_pot_k41,
        "E_tot": E_tot
    }

from jax.lax import scan

N_iter = 1000

# ground_state_rb87, energies = scan(
#     f=imaginary_time_step_single_species,
#     init=init_state_rb87.into(space="freq"),
#     xs=None,
#     length=N_iter
# )
# generate_panel_plot(ground_state_rb87)

ground_states, energies = scan(
    f=imaginary_time_step_dual_species,
    init=(init_state_rb87.into(space="freq"), init_state_k41.into(space="freq")),
    xs=None,
    length=N_iter
)

energies["E_kin"] = energies["E_kin_rb87"] + energies["E_kin_k41"]
energies["E_pot"] = energies["E_pot_rb87"] + energies["E_pot_k41"]

ground_state_rb87, ground_state_k41 = ground_states

plot_fftarray(ground_state_rb87)
plot_fftarray(ground_state_k41)

# %%

# %%
# import numpy as np
# from bokeh.plotting import figure, show
# # plot the energy trend during the imaginary time evolution
# plt = figure(
#     width=800, height=400, min_border=50,
#     y_axis_type="log",
#     title="Energy values during imaginary time evolution",
#     x_axis_label="Iteration step",
#     y_axis_label="Energy in µK"
# )
# x_num_iter = np.arange(N_iter)
# # numerical solution
# plt.line(
#     x_num_iter, energies["E_kin"],
#     line_width=1.5, color="red", legend_label="Kinetic Energy"
# )
# plt.line(
#     x_num_iter, energies["E_pot"],
#     line_width=1.5, color="green", legend_label="Potential Energy"
# )
# plt.line(
#     x_num_iter, energies["E_tot"],
#     line_width=1.5, color="blue", legend_label="Total Energy"
# )

# # show the plot
# # show(plt)

# pn.io.push_notebook(plt)

import numpy as np
import panel as pn
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

# Create some example data
# N_iter = 100
# energies = {
#     "E_kin": np.random.rand(N_iter),
#     "E_pot": np.random.rand(N_iter),
#     "E_tot": np.random.rand(N_iter)
# }
data = pd.DataFrame({
    'x': np.arange(N_iter),
    'E_kin': energies["E_kin"],
    'E_pot': energies["E_pot"],
    'E_tot': energies["E_tot"]
})

def create_bokeh_plot(data):
    plt = figure(
        width=800, height=400, min_border=50,
        y_axis_type="log",
        title="Energy values during imaginary time evolution",
        x_axis_label="Iteration step",
        y_axis_label="Energy in µK"
    )

    source_kin = ColumnDataSource(data={'x': data['x'], 'y': data['E_kin']})
    source_pot = ColumnDataSource(data={'x': data['x'], 'y': data['E_pot']})
    source_tot = ColumnDataSource(data={'x': data['x'], 'y': data['E_tot']})

    plt.line('x', 'y', source=source_kin, line_width=1.5, color="red", legend_label="Kinetic Energy")
    plt.line('x', 'y', source=source_pot, line_width=1.5, color="green", legend_label="Potential Energy")
    plt.line('x', 'y', source=source_tot, line_width=1.5, color="blue", legend_label="Total Energy")

    plt.legend.title = 'Energy Type'
    return plt

# Wrap the Bokeh plot in a function
def bokeh_plot():
    plot = create_bokeh_plot(data)
    return plot

# Create a Panel object
bokeh_pane = pn.pane.Bokeh(bokeh_plot())

# Serve the Panel app
bokeh_pane.servable()


# %%



