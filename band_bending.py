import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import constants as const


def gauss_dos(energy, shift=0, sigma=0.25):
    """Gaussian density of states"""
    return np.exp(-(energy - shift) ** 2 / (sigma ** 2))


def fd(en, mu, tempr):
    """Fermi-Dirac distribution function"""

    return 1.0 / (np.exp((en - mu) * const.el / (const.kb * tempr)) + 1)


def poisson_solver(rho, boundary1, boundary2, delta):
    rho /= const.eps0
    num_el = len(rho)
    pot = np.tril(np.ones(num_el), 1) + np.triu(np.ones(num_el), -1) - 1 - 3 * np.diag(np.ones(num_el))
    rho = rho * delta**2

    if boundary1[0] == 'd':
        pot[0, 0] = 1
        pot[0, 1] = 0
        rho[0] = -boundary1[1]
    else:
        pot[0, 0] = -1
        pot[0, 1] = 1
        rho[0] = boundary1[1] * delta

    if boundary2[0] == 'd':
        pot[num_el - 1, num_el - 1] = 1
        pot[num_el - 1, num_el - 2] = 0
        rho[num_el - 1] = -boundary2[1]
    else:
        pot[num_el - 1, num_el - 1] = -1
        pot[num_el - 1, num_el - 2] = 1
        rho[num_el - 1] = -boundary2[1] * delta

    return np.linalg.inv(pot) @ -rho


def poisson_solver1(rho, boundary1, boundary2, delta):
    num_el = len(rho)
    pot = np.tril(np.ones(num_el), 1) + np.triu(np.ones(num_el), -1) - 1 - 3 * np.diag(np.ones(num_el))
    rho = rho * delta**2

    if boundary1[0] == 'd':
        pot[0, 0] = -2
        pot[0, 1] = 1
        rho[0] -= boundary1[1]
    else:
        pot[0, 0] = -1
        pot[0, 1] = 1
        rho[0] = 0.5 * rho[0] - boundary1[1] * delta

    if boundary2[0] == 'd':
        pot[num_el - 1, num_el - 1] = -2
        pot[num_el - 1, num_el - 2] = 1
        rho[num_el - 1] -= boundary2[1]
    else:
        pot[num_el - 1, num_el - 1] = -1
        pot[num_el - 1, num_el - 2] = 1
        rho[num_el - 1] = 0.5 * rho[num_el - 1] - boundary2[1] * delta

    return np.linalg.inv(pot) @ -rho


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
        e_length = 2000

        en = np.linspace(e_min, self.ef, e_length)
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

    plt.plot(energy, dos)
    plt.show()

    layers = LayersStack()
    layers.initialize_from_dos(energy, dos, 30)
    layers.set_potential(np.linspace(4, -4, layers.num_layers))

    density = layers.density()
    plt.plot(density, '.-')
    plt.show()

    ldos = layers.ldos(energy)
    plt.contourf(ldos)
    plt.show()

    l = Layer(energy, dos)
    print(l.density())

    interp = interpolate.interp1d(energy, dos)
    layers = [Layer(energy, interp, z=j) for j in range(10)]

    energy1 = np.linspace(-25, 25, 300)
    energy2 = np.linspace(-25, 25, 500)

    plt.plot(energy, dos, energy1, l.dos(energy1), energy2, l1.dos(energy2))
    plt.show()

    z = np.linspace(0, 50, 2000)
    rho = np.zeros(z.shape)

    for j in range(1, 40):
        print(j)
        rho += gauss_dos(z, j + 2, 0.15)

    rho = (-gauss_dos(z, 20, 1) + gauss_dos(z, 30, 1)) * const.el / 1e-5
    pot = poisson_solver(rho, ('d', 0), ('n', 0), 0.1)
    # pot1 = poisson_solver1(rho, ('d', 0), ('n', 0), 0.1)

    plt.plot(z, rho)
    plt.plot(z, pot)
    # plt.plot(z, pot1, 'o')
    plt.show()

def main1():

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



if __name__ == '__main__':

    main1()