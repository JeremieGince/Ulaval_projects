import numpy
from scipy import constants

mu0: float = 1.2566e-6  # [kg m A^−2 s^−2]
c: float = 2.998e8  # [m/s]
alpha: float = 1/137.035999
hbar: float = 1.054571817e-34  # [j*s]
h: float = 6.602607015e-34  # [j*s]
Z_H: int = 1
q_e: float = 1.6021766208e19  # [As]
m_e: float = 9.109e-31  # [kg]
m_n: float = 1.6889e-27  # [kg]
m_p: float = 1.6725e-27  # [kg]
mu_B: float = (q_e * hbar)/(2 * m_e)
g_ell: float = 1.0
g_s: float = 2.0
s_H: numpy.ndarray = numpy.array([1/2])  # possible s for Hydrogen
k_B: float = constants.value("Stefan-Boltzmann constant")  # [W m^-2 K^-4] Boltzmann constant
# a0 = (hbar**2)/(m_e*(q_e**2))
a0: float = constants.value("Bohr radius")


def mu_mag(L, S) -> float:
    """
    return the magnetic field with quantum number L and S
    :param L: quantume number L (float)
    :param S: quantum number S (S)
    :return: (float)
    """
    return -(mu_B/hbar)*(g_ell*L + g_s*S)


def mu_mass(m_1, m_2) -> float:
    """
    return the reduced mass
    :param m_1: mass 1 (float)
    :param m_2: mass 2 (float)
    :return: reduced mass (float)
    """
    return (m_1*m_2)/(m_1+m_2)


mu_H = mu_mass(m_p, m_e)  # reduced mass of the hydrogen


if __name__ == '__main__':
    print(f"alpha = {alpha}, \n"
          f"hbar = {hbar}, \n"
          f"c = {c}, \n"
          f"a0 = {a0}, \n"
          f"mu_H = {mu_H}")
