import numpy as np
import constants as const


def gauss_dos(energy, shift=0, sigma=0.25):
    """Gaussian density of states"""
    return np.exp(-(energy - shift) ** 2 / (sigma ** 2))


def fd(en, mu, tempr):
    """Fermi-Dirac distribution function"""

    return 1.0 / (np.exp((en - mu) * const.el / (const.kb * tempr)) + 1)