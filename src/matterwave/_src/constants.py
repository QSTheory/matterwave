from typing import Protocol
from scipy.constants import pi, hbar, atomic_mass

class AtomicSpecies(Protocol):

    wavelength: float # m
    """Optical transition wavelength in m."""

    mass: float # kg
    """Mass in kg."""

    @property
    def wavenumber(self) -> float:
        """Wavenumber for the given wavelength: `2*pi/wavelength`"""
        return 2*pi / self.wavelength

    @property
    def hbark(self) -> float:
        """Recoil momentum for the given wave number k: `hbar*k`"""
        return hbar * self.wavenumber

    @property
    def vr(self) -> float:
        """Recoil velovity for the given wavelength and mass: `hbar*k/mass`"""
        return self.hbark/self.mass


class Rubidium87(AtomicSpecies):
    """Rubidium-87 D2 transition optical properties. Numbers are taken from Ref.
    [1]_.

    References
    ----------
    .. [1] Daniel A. Steck, "Rubidium 87 D Line Data", available online at
        http://steck.us/alkalidata (revision 2.3.3, 28 May 2024).
    """
    mass = 86.909 * atomic_mass
    wavelength = 780.032e-9
