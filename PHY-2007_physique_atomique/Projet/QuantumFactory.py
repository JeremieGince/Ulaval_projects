import Constantes as const
import numpy as np
import sympy as sp
import mpmath
import scipy as sc
from scipy import integrate


class QuantumFactory:
    """
    QuantumFactory is a module to combine a bunch of static method util in quantum mechanics
    """
    @staticmethod
    def zeta_n(n=sp.Symbol("n", real=True), r=sp.Symbol("r", real=True),
               z=sp.Symbol("Z", real=True), mu=sp.Symbol('mu', real=True)):
        """
        return the zeta_n function
        :param n: orbital number n (int)
        :param r: rayon variable (sympy object)
        :param z: (int)
        :param mu: reduced mass
        :return: zeta_n function (sympy object)
        """
        return (2 * z * const.alpha * mu * const.c * r) / (n * const.hbar)

    @staticmethod
    def u_n_ell(n=sp.Symbol("n", real=True), ell=sp.Symbol("ell", real=True),
                z=sp.Symbol("Z", real=True), mu=sp.Symbol('mu', real=True)):
        """
        return the u_{n \ell} function
        :param n: orbital number n (int)
        :param ell: kinetic momentum  (int)
        :param z: (int)
        :param mu: reduced mass (float)
        :return: (sympy object)
        """
        u_coeff_norm = sp.sqrt(((2 * z * const.alpha * mu * const.c) / (n * const.hbar)) ** 3)
        u_coeff_fact = sp.sqrt(np.math.factorial(n - ell - 1) / (2 * n * np.math.factorial(n + ell)))
        u_coeff = u_coeff_norm * u_coeff_fact

        exp_term = sp.exp(-QuantumFactory.zeta_n(n=n, z=z, mu=mu) / 2)
        laguerre_term = sp.assoc_laguerre(n - ell - 1, 2 * ell + 1, QuantumFactory.zeta_n(n=n, z=z, mu=mu))

        return u_coeff * exp_term * (QuantumFactory.zeta_n(n=n, z=z, mu=mu) ** ell) * laguerre_term

    @staticmethod
    def Y_ell_m_ell(ell: int, m_ell: int):
        """
        Return the spherical harmonic function
        :param ell: kinetic momentum (int)
        :param m_ell:
        :return: (sympy object)
        """
        from sympy.functions.special.spherical_harmonics import Ynm
        theta, phi = sp.Symbol("theta", real=True), sp.Symbol("phi", real=True)
        return sp.FU['TR8'](Ynm(ell, m_ell, theta, phi).expand(func=True))

    @staticmethod
    def get_valid_ell_with_n(n: int):
        return np.array([i for i in np.arange(start=0, stop=n, step=1)])

    @staticmethod
    def get_valid_m_ell_with_ell(ell: int):
        return np.array([i for i in np.arange(start=-ell, stop=ell+1, step=1)])

    @staticmethod
    def get_valid_m_s_with_s(s: float):
        return np.array([i for i in np.arange(start=-s, stop=s+1, step=1)])

    @staticmethod
    def get_valid_transitions_n_to_n(n, n_prime) -> list:
        """
        Get a Transitions(list) container of all of the valid transition of n to n_prime
        :param n: (int)
        :param n_prime: (int)
        :return: Transitions(list) of Transition object of the initial state and final state (Transitions)
        """
        from Transition import Transition
        from Transitions import Transitions
        import warnings
        warnings.warn("Warning! This method seems to be not valid", DeprecationWarning)
        valid_transitions = Transitions()  # must be a Transitions object
        for init_quantum_state in QuantumFactory.get_valid_quantum_state_for_n(n):
            for end_quantum_state in init_quantum_state.get_valid_transitions_state_to_n(n_prime):
                valid_transitions.append(Transition(init_quantum_state, end_quantum_state))
        return valid_transitions

    @staticmethod
    def get_valid_transitions_n_to_n_prime(n: int, n_prime: int) -> list:
        """
        Get a Transitions(list) container of all of the valid transition of n to n_prime
        :param n: (int)
        :param n_prime: (int)
        :return: Transitions(list) of Transition object of the initial state and final state (Transitions)
        """
        from Transition import Transition
        from Transitions import Transitions
        valid_transitions: Transitions = Transitions()
        for init_quantum_state in QuantumFactory.get_valid_quantum_state_for_n(n):
            for end_quantum_state in QuantumFactory.get_valid_quantum_state_for_n(n_prime):
                if Transition.possible(init_quantum_state, end_quantum_state):
                    valid_transitions.append(Transition(init_quantum_state, end_quantum_state))
        return valid_transitions

    @staticmethod
    def get_valid_quantum_state_for_n(n, s_array: np.ndarray = const.s_H) -> np.ndarray:
        """
        Get all the valid quantum state for the orbital number n
        :param n: orbital number n (int)
        :param s_array: array of possible spin number s (numpy.ndarray)
        :return: array of valid quantum state (numpy.ndarray[QuantumState])
        """
        from QuantumState import QuantumState
        valid_states: list = list()
        for ell in QuantumFactory.get_valid_ell_with_n(n):
            for m_ell in QuantumFactory.get_valid_m_ell_with_ell(ell):
                for s in s_array:
                    for m_s in QuantumFactory.get_valid_m_s_with_s(s):
                        valid_states.append(QuantumState(n, ell, m_ell, s, m_s))
        return np.array(valid_states)

    @staticmethod
    def get_g_n(n: int) -> int:
        """
        Get the degenesrence number of the level n
        :param n: orbital level (int)
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

    @staticmethod
    def get_state_energy_unperturbeted(n: int, z=sp.Symbol("Z", real=True), mu=sp.Symbol('mu', real=True)):
        """
        Get the energy of the current quantum state
        :param n:
        :param z: electric charge
        :param mu:
        :return: the energy of the current quantum state (float if z and mu are float else sympy object)
        """
        numerator = - (z**2)*(const.alpha**2)*mu*(const.c**2)
        denumerator = 2*(n**2)
        return numerator/denumerator

    @staticmethod
    def get_delta_energy_unpertuberted(n: int, n_prime: int,
                                       z=sp.Symbol('Z', real=True), mu=sp.Symbol('mu', real=True)):
        """
        Getter of the transition energy without any pertubation
        :param n: initial orbital number n (int)
        :param n_prime: final orbital number n (int)
        :param z: (float)
        :param mu: reduced mass (float)
        :return: transition energy (float) or transition energy (sympy object)
        """
        e = QuantumFactory.get_state_energy_unperturbeted(n, z, mu)
        e_prime = QuantumFactory.get_state_energy_unperturbeted(n_prime, z, mu)
        return e - e_prime

    @staticmethod
    def get_transition_angular_frequency_unperturbated(n: int, n_prime: int,
                                                       z=sp.Symbol('Z', real=True), mu=sp.Symbol('mu', real=True)):
        """
        Getter of the transition angular frequency without any pertubation
        :param n: initial orbital number n (int)
        :param n_prime: final orbital number n (int)
        :param z: (float)
        :param mu: reduced mass (float)
        :return: angular frequency (float) or angular frequency (sympy object)
        """
        return QuantumFactory.get_delta_energy_unpertuberted(n, n_prime, z, mu)/const.hbar

    @staticmethod
    def decay_number(n: int, T: float, z: int = const.Z_H, mu: float = const.mu_H):
        """
        Return the decay number of the level n for unperturbeted energy.
        :param n: orbital number n (int)
        :param T: Current temperature (float)
        :param z: (int)
        :param mu: reduced mass (float)
        :return: a sympy expression of the decay number (sympy object)
        """
        alpha = sp.Symbol('alpha')  # proportional function
        g = QuantumFactory.get_g_n(n)
        return alpha*g*sp.exp(-QuantumFactory.get_state_energy_unperturbeted(n, z, mu)/(const.k_B*T))

    @staticmethod
    def decay_number_ratio(n: int, n_prime: int, k_B: float, T: float, z: int = const.Z_H, mu: float = const.mu_H):
        """
        Return the ratio of decay number of the levels n to n_prime for unperturbeted energy.
        :param n: initial orbital number n (int)
        :param n_prime: final orbital number n (int)
        :param k_B: (float)
        :param T: Current temperature (float)
        :param z: (int)
        :param mu: reduced mass (float)
        :return: the ration of decay number (float)
        """
        N_i = QuantumFactory.decay_number(n, k_B, T, z, mu)
        N_j = QuantumFactory.decay_number(n_prime, k_B, T, z, mu)
        return (N_i/N_j).evalf()

    @staticmethod
    def intensity_of_the_beam(n: int, n_prime: int, T: float, z: int=const.Z_H, mu: float=const.mu_H):
        transitions = Transitions(n, n_prime)
        alpha = sp.Symbol('alpha')  # proportional function
        omega = QuantumFactory.get_transition_angular_frequency_unperturbated(n, n_prime, z, mu)
        N = QuantumFactory.decay_number(n, T, z, mu)
        Rs_mean = transitions.get_spontanious_decay_mean(z, mu)
        return alpha*N*omega*Rs_mean

    @staticmethod
    def ratio_intensity_of_the_beam(n: int, n_prime: int, i: int, j: int,
                                    T: float, z: int = const.Z_H, mu: float = const.mu_H):
        I1 = QuantumFactory.intensity_of_the_beam(n, n_prime, T, z, mu)
        I2 = QuantumFactory.intensity_of_the_beam(i, j, T, z, mu)
        return (I1/I2).evalf()

    @staticmethod
    def bracket_product(wave_function, wave_function_prime, operator=None, algo="scipy"):
        algos = {"sympy": QuantumFactory.bracket_product_sympy,
                 "scipy":QuantumFactory.bracket_product_scipy}
        assert algo in algos.keys()
        return algos[algo](wave_function, wave_function_prime, operator)


    @staticmethod
    def bracket_product_sympy(wave_function, wave_function_prime, operator=None):
        r, theta, phi = sp.Symbol("r", real=True), sp.Symbol("theta", real=True), sp.Symbol("phi", real=True)
        jacobian = (r**2) * sp.sin(theta)
        integral_core = sp.conjugate(wave_function) * (operator if operator is not None else 1) * wave_function_prime
        integral_core_expension = sp.expand(sp.FU["TR8"](jacobian * integral_core), func=True).simplify()

        # creation of the Integral object and first try to resolve it
        bracket_product = sp.Integral(integral_core_expension,
                                      (phi, 0, 2 * mpmath.pi), (r, 0, mpmath.inf), (theta, 0, mpmath.pi))
        # print(f"\n Integral bracket_product: {bracket_product}")

        # simplify the result of the first try and evaluation of the integral, last attempt
        bracket_product = bracket_product.simplify().evalf(n=50, maxn=3_000, strict=True)
        return bracket_product

    @staticmethod
    def bracket_product_scipy(wave_function, wave_function_prime, operator=None):
        r, theta, phi = sp.Symbol("r", real=True), sp.Symbol("theta", real=True), sp.Symbol("phi", real=True)
        jacobian = (r ** 2) * sp.sin(theta)
        integral_core = sp.conjugate(wave_function) * (operator if operator is not None else 1) * wave_function_prime

        integral_core_expension = sp.expand(sp.FU["TR8"](jacobian*integral_core), func=True).simplify()

        def bound_r(_):
            return [0, mpmath.inf]

        def bound_phi(_, __):
            return [0, 2 * mpmath.pi]

        def bound_theta():
            return [0, mpmath.pi]

        # Process the integral with scipy
        integral_core_lambdify = sp.lambdify((theta, r, phi), integral_core_expension, modules="numpy")
        bracket_product = sc.integrate.nquad(integral_core_lambdify, [bound_phi, bound_r, bound_theta])[0]
        return bracket_product


if __name__ == '__main__':
    from Transitions import Transitions
    from QuantumState import QuantumState

    qs1 = QuantumState(n=2, ell=1, m_ell=0, s=0.5, m_s=0.5)
    qs2 = QuantumState(n=1, ell=0, m_ell=0, s=0.5, m_s=0.5)

    print(QuantumFactory.bracket_product(qs1.get_wave_fonction(z=const.Z_H, mu=const.mu_H), qs2.get_wave_fonction(z=const.Z_H, mu=const.mu_H)))
