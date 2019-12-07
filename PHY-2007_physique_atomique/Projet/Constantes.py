
mu0 = 1.2566e-6  # [kg m A^−2 s^−2]
c = 2.998e8  # [m/s]
alpha = 1/137.035999
hbar = 1.054571817e-34  # [j*s]
h = 6.602607015e-34  # [j*s]
Z_H = 1
q_e = 1.6021766208e19  # [As]
m_e = 9.109e-31  # [kg]
m_n = 1.6889e-27  # [kg]
m_p = 1.6725e-27  # [kg]
mu_B = (q_e * hbar)/(2 * m_e)
g_ell = 1
g_s = 2


def mu_mag(L, S):
    return -(mu_B/hbar)*(g_ell*L + g_s*S)


def mu_mass(m_1, m_2):
    return (m_1*m_2)/(m_1+m_2)


mu_H = mu_mass(m_p, m_e)
