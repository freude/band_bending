import numpy as np
import constants as const

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