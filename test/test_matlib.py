import numpy as np
from layers import MatLib, gauss_dos
import unittest
from scipy import interpolate


class MyTestCase(unittest.TestCase):

    def setUp(self):
        energy = np.linspace(-15, 15, 2000)
        self.dos = gauss_dos(energy, -3)
        self.matlib_gaas = MatLib(GaAs=[energy, self.dos])
        self.matlib_empty = MatLib()

    def test_type_gaas(self):
        self.assertIsInstance(self.matlib_gaas['GaAs'], interpolate._interpolate.interp1d)


if __name__ == '__main__':

    unittest.main()