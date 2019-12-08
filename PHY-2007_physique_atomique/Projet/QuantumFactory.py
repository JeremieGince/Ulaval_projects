import Constantes as const
import numpy as np
import sympy as sp


class QuantumFactory:
    """
    QuantumFactory is a module to combine a bunch of static method util in quantum mechanics
    """
    @staticmethod
    def xi_n(n=sp.Symbol("n", real=True), r=sp.Symbol("r", real=True),
             z=sp.Symbol("Z", real=True), mu=sp.Symbol('mu', real=True)):
        """

        :param n:
        :param r:
        :param z: (float)
        :param mu: reduced mass
        :return:
        """
        return (2 * z * const.alpha * mu * const.c * r) / (n * const.hbar)

    @staticmethod
    def u_n_ell(n=sp.Symbol("n", real=True), ell=sp.Symbol("ell", real=True),
                z=sp.Symbol("Z", real=True), mu=sp.Symbol('mu', real=True)):
        """

        :param n:
        :param ell:
        :param z:
        :param mu: reduced mass (float)
        :return:
        """
        u_coeff_norm = sp.sqrt(((2 * z * const.alpha * mu * const.c) / (n * const.hbar)) ** 3)
        u_coeff_fact = sp.sqrt(np.math.factorial(n - ell - 1) / (2 * n * np.math.factorial(n + ell)))
        u_coeff = u_coeff_norm * u_coeff_fact

        exp_term = sp.exp(-QuantumFactory.xi_n(n=n, z=z, mu=mu) / 2)
        laguerre_term = sp.assoc_laguerre(n - ell - 1, 2 * ell + 1, QuantumFactory.xi_n(n=n, z=z, mu=mu))

        return u_coeff * exp_term * (QuantumFactory.xi_n(n=n, z=z, mu=mu) ** ell) * laguerre_term

    @staticmethod
    def get_valid_ell_with_n(n: int):
        return np.array([i for i in range(0, n)])

    @staticmethod
    def get_valid_m_ell_with_ell(ell: int):
        return np.array([i for i in range(-ell, ell+1)])

    @staticmethod
    def get_valid_m_s_with_s(s: float):
        return np.array([i for i in np.arange(start=-s, stop=s+1, step=1)])

    @staticmethod
    def get_valid_quantum_state_for_n(n) -> list:
        from QuantumState import QuantumState
        valid_states = list()
        for ell in QuantumFactory.get_valid_ell_with_n(n):
            for m_ell in QuantumFactory.get_valid_m_ell_with_ell(ell):
                for s in [1/2]:
                    for m_s in QuantumFactory.get_valid_m_s_with_s(s):
                        valid_states.append(QuantumState(n, ell, m_ell, s, m_s))
        return valid_states

    @staticmethod
    def get_valid_transitions_n_to_n(n, n_prime) -> list:
        """
        Get a list of all of the valid transition of n to n_prime
        :param n: (int)
        :param n_prime: (int)
        :return: list of Transition object of the initial state and final state
        """
        from Transition import Transition
        valid_transitions = list()  # must be a Transitions object
        for init_quantum_state in QuantumFactory.get_valid_quantum_state_for_n(n):
            for end_quantum_state in init_quantum_state.get_valid_transitions_state_to_n(n_prime):
                valid_transitions.append(Transition(init_quantum_state, end_quantum_state))
        return valid_transitions

    @staticmethod
    def get_g_n(n: int) -> int:
        """
        Get the degenesrence number of the level n
        :param n: orbital level
        :return: degenesrecence number (int)
        """
        deg_dico = dict()
        for quantum_state in QuantumFactory.get_valid_quantum_state_for_n(n):
            energy = quantum_state.get_state_energy()
            if energy in deg_dico.keys():
                deg_dico[energy] += 1
            else:
                deg_dico[energy] = 1
        return int(np.sum(list(deg_dico.values())))