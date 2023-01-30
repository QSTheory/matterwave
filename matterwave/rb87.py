"""Constants of Rubidium 87"""
from scipy.constants import hbar
from numpy import pi

m: float = 86.909 * 1.66053906660e-27
"""The atom's mass in kg."""

lambda_L: float = 780 * 1e-9
"""The :math:`D_2` transition wavelength :math:`\lambda_L` in m."""

k_L: float = 2 * pi / lambda_L
""":math:`k_L = \\frac{2\pi}{\lambda_L}`."""

#These two are equivalent
# hbark = h / lambda_L

hbark: float = hbar * k_L
""":math:`\hbar k_L = \\frac{h}{\lambda_L}`."""

hbarkv: float = hbark/m
""":math:`\\frac{\hbar k_L}{m}`."""
