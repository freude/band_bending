import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import constants as const
from aux_functions import fd, gauss_dos
from poisson_solvers import poisson_solver


class Layer(object):
    """
    Class defines a single layer that is capable of charge exchange
    """
    matlib = None

    def __init__(self, **kwargs):

        if Layer.matlib is None:
            raise NotImplementedError("There is no materials registered.")

        self.pot = kwargs.get('pot', 0.0)
        self.tempr = kwargs.get('tempr', 300)
        self.ef = kwargs.get('ef', 0.0)
        self.z = kwargs.get('z', 0.0)
        self.material = kwargs.get('mat', None)

    def dos(self, energy):
        """DOS"""

        return self.matlib[self.material](energy - self.pot)

    def density(self):
        """Density"""

        e_min = -15
        e_max = self.ef + 5 * const.kb * self.tempr / const.el
        e_length = 2000

        en = np.linspace(e_min, e_max, e_length)
        return np.trapz(self.dos(en - self.pot) * fd(en, self.ef, self.tempr), en)


class LayersStack(object):
    """
    Class defines a stack layer
    """

    def __init__(self, **kwargs):

        self.ef = kwargs.get('ef', 0.0)
        self.tempr = kwargs.get('tempr', 300)

        layers = kwargs.get('layers', [])
        self.layers = [Layer(mat=item) for item in layers]

        self.num_layers = len(self.layers)

    def density(self):

        ans = [self.layers[j].density() for j in range(self.num_layers)]

        return np.array(ans)

    def ldos(self, energy):

        ldos = []

        for j, item in enumerate(self.layers):
            ldos.append(item.dos(energy))

        return np.array(ldos).T

    def __add__(self, other):

        return LayersStack(layers=self.layers + other.layers, tempr=self.tempr, ef=self.ef)

    def set_potential(self, pot):

        for j, item in enumerate(self.layers):
            item.pot = pot[j]

    def get_potential(self):

        pot = []

        for j, item in enumerate(self.layers):
            pot.append(item.pot)

        return np.array(pot)

    def iterate(self):

        citeria = 1e-5
        converged = False

        if not converged:
            rho = self.get_charges()
            pot = poisson_solver(rho, ('d', 0), ('n', 0), 0.1)

    def poisson(self):
        rho = self.density()
        pot = poisson_solver(rho, ('d', 0), ('n', 0), 0.1)
        return pot


class MatLib(dict):

    def __init__(self, **kwargs):

        for key, value in kwargs.items():
            kwargs[key] = self.interp(value)

        super(MatLib, self).__init__(**kwargs)
        Layer.matlib = self

    def __setitem__(self, key, value):

        super().__setitem__(key, self.interp(value))
        Layer.matlib = self

    @staticmethod
    def interp(value):
        if isinstance(value, list):
            value = interpolate.interp1d(value[0], value[1], bounds_error=False, fill_value=0)
            return value
        elif isinstance(value, interpolate._interpolate.interp1d):
            return value
        else:
            raise TypeError("MatLib values can be only arrays or 1D interpolants")


def main():

    energy = np.linspace(-15, 15, 2000)
    dos = gauss_dos(energy, -3) + gauss_dos(energy, -1) + gauss_dos(energy, 1) + gauss_dos(energy, 3)

    # register materials and their DOS
    MatLib(Tet=[energy, dos])
    layers = LayersStack(layers=['Tet']*100)
    print(layers.num_layers)

    layers.set_potential(np.linspace(4, -4, layers.num_layers))

    density = layers.density()
    plt.plot(density, '.-')
    plt.show()

    ldos = layers.ldos(energy)
    plt.contourf(ldos)
    plt.show()

    plt.plot(layers.get_potential())
    plt.show()


def main1():

    energy = np.linspace(-15, 15, 2000)
    dos = gauss_dos(energy, -1) + gauss_dos(energy, 1)
    dos1 = gauss_dos(energy, -2.8) + gauss_dos(energy, -0.8)

    # register materials and their DOS
    MatLib(Tet=[energy, dos], Tet1=[energy, dos1])
    layers = LayersStack(layers=['Tet']*10+['Tet1']*10+['Tet']*10+['Tet1']*10+['Tet']*10)
    # layers.set_potential(np.linspace(4, -4, layers.num_layers))

    ldos = layers.ldos(energy)
    plt.contourf(ldos)
    plt.show()

    pot = layers.poisson()

    plt.plot(pot)
    plt.show()


if __name__ == '__main__':

    main1()