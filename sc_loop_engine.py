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