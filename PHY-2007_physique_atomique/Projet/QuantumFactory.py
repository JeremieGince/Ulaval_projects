import Constants as const
import numpy as np
import sympy as sp
import mpmath
import scipy as sc
from scipy import integrate
import mcint
import random
import itertools
import numba
from numba import cuda, prange


class QuantumFactory:
    """
    QuantumFactory is a module to combine a bunch of static method util in quantum mechanics
    """
    @staticmethod
    def zeta_n(n=sp.Symbol("n", real=True), r=sp.Symbol("r", real=True, positive=True),
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
        # return sp.FU['TR8'](Ynm(ell, m_ell, theta, phi).expand(func=True))
        # return Ynm(ell, m_ell, theta, phi)
        return Ynm(ell, m_ell, theta, phi).expand(func=True)

    @staticmethod
    def Y_ell_m_ell_scipy(ell, m_ell):
        pass

    @staticmethod
    def get_hydrogen_wave_function(n: int, ell: int, m_ell: int):
        """
        Give the hydrogen wave function of sympy
        :param n: quantum number n (int)
        :param ell: quantum number ell (int)
        :param m_ell: quantum number m_ell (int)
        :return:
        """
        from sympy.physics import hydrogen
        r, theta, phi = sp.Symbol("r", real=True, positive=True),\
                        sp.Symbol("theta", real=True), sp.Symbol("phi", real=True)
        return hydrogen.Psi_nlm(n, ell, m_ell, r, phi, theta, Z=1/const.a0)

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
    def get_valid_transitions_n_to_n_prime(n: int, n_prime: int, hydrogen: bool = False) -> list:
        """
        Get a Transitions(list) container of all of the valid transition of n to n_prime
        :param n: initial quantum number n (int)
        :param n_prime: final quantum number n (int)
        :param hydrogen: if we want to cast quantum state in quantum hydrogen state (bool)
        :return: Transitions(list) of Transition object of the initial state and final state (Transitions)
        """
        from Transition import Transition
        from Transitions import Transitions
        valid_transitions: Transitions = Transitions()
        for init_quantum_state in QuantumFactory.get_valid_quantum_state_for_n(n, hydrogen=hydrogen):
            for end_quantum_state in QuantumFactory.get_valid_quantum_state_for_n(n_prime, hydrogen=hydrogen):
                if Transition.possible(init_quantum_state, end_quantum_state):
                    valid_transitions.append(Transition(init_quantum_state, end_quantum_state))
        return valid_transitions

    @staticmethod
    def get_valid_quantum_state_for_n(n, s_array: np.ndarray = const.s_H, hydrogen: bool = False) -> np.ndarray:
        """
        Get all the valid quantum state for the orbital number n
        :param n: orbital number n (int)
        :param s_array: array of possible spin number s (numpy.ndarray)
        :param hydrogen: if we want to cast quantum state in quantum hydrogen state (bool)
        :return: array of valid quantum state (numpy.ndarray[QuantumState])
        """
        from QuantumState import QuantumState
        valid_states: list = list()
        for ell in QuantumFactory.get_valid_ell_with_n(n):
            for m_ell in QuantumFactory.get_valid_m_ell_with_ell(ell):
                for s in s_array:
                    for m_s in QuantumFactory.get_valid_m_s_with_s(s):
                        valid_states.append(QuantumState(n, ell, m_ell, s, m_s, hydrogen))
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
        g = QuantumFactory.get_g_n(n)
        return g*sp.exp(-QuantumFactory.get_state_energy_unperturbeted(n, z, mu)/(const.k_B*T))

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
        from Transitions import Transitions
        transitions = Transitions(n, n_prime)
        alpha = sp.Symbol('alpha')  # proportional function
        omega = QuantumFactory.get_transition_angular_frequency_unperturbated(n, n_prime, z, mu)
        N = QuantumFactory.decay_number(n, T, z, mu)
        Rs_mean = transitions.get_spontanious_decay_mean(z, mu)
        return alpha*N*omega*Rs_mean

    @staticmethod
    def relative_intensity_of_the_beam(n: int, n_prime: int, i: int, j: int,
                                       T: float, z: int = const.Z_H, mu: float = const.mu_H):
        I1 = QuantumFactory.intensity_of_the_beam(n, n_prime, T, z, mu)
        I2 = QuantumFactory.intensity_of_the_beam(i, j, T, z, mu)
        return (I1/I2).evalf()

    @staticmethod
    def bracket_product(wave_function, wave_function_prime, operator=None, algo="mcint"):
        """
        Call the algo function to make the scalar product
        :param wave_function: bra
        :param wave_function_prime: ket
        :param operator: operator
        :param algo: algo to use to compute the integral (str) element of
                     {sympy, sympy_ns, scipy_nquad, scipy_tplquad, mcint}
        :return: the result of algo(wave_function, wave_function_prime, operator)
        """
        algos = {"sympy": QuantumFactory.bracket_product_sympy,
                 "sympy_ns": QuantumFactory.bracket_product_sympy_ns,
                 "scipy_nquad": QuantumFactory.bracket_product_scipy_nquad,
                 "scipy_tplquad": QuantumFactory.bracket_product_scipy_tplquad,
                 "mcint": QuantumFactory.bracket_product_mcint}
        assert algo in algos.keys()
        return algos[algo](wave_function, wave_function_prime, operator)

    @staticmethod
    def bracket_product_sympy(wave_function, wave_function_prime, operator=None):
        """

        :param wave_function:
        :param wave_function_prime:
        :param operator:
        :return:
        """
        r, theta, phi = sp.Symbol("r", real=True, positive=True), sp.Symbol("theta", real=True), sp.Symbol("phi", real=True)
        jacobian = (r**2) * sp.sin(theta)
        integral_core = sp.conjugate(wave_function) * (operator if operator is not None else 1) * wave_function_prime
        integral_core_expension = sp.expand(sp.FU["TR8"](jacobian*integral_core), func=True).simplify()

        # creation of the Integral object and first try to resolve it
        bracket_product = sp.Integral(integral_core_expension,
                                      (phi, 0, 2 * mpmath.pi), (r, 0, mpmath.inf), (theta, 0, mpmath.pi),
                                      risch=False).doit()

        # simplify the result of the first try and evaluation of the integral, last attempt
        bracket_product = bracket_product.simplify().evalf(n=50, maxn=3_000, strict=True)
        return bracket_product

    @staticmethod
    def bracket_product_sympy_ns(wave_function, wave_function_prime, operator=None):
        """
        Compute the scalar product <bra|operator|ket> without simplification to use the symbolic integration of sympy
        :param wave_function: bra
        :param wave_function_prime: ket
        :param operator: operator
        :return: bracket product (float)
        """
        r, theta, phi = sp.Symbol("r", real=True, positive=True),\
                        sp.Symbol("theta", real=True), sp.Symbol("phi", real=True)
        jacobian = (r ** 2) * sp.sin(theta)
        integral_core = wave_function * (operator if operator is not None else 1) * sp.conjugate(wave_function_prime)

        # creation of the Integral object and first try to resolve it
        bracket_product = sp.integrate(jacobian*integral_core,
                                       (phi, 0, 2 * np.pi), (r, 0, np.inf), (theta, 0, np.pi),
                                       risch=False)

        # simplify the result of the first try and evaluation of the integral, last attempt
        bracket_product = bracket_product.simplify().evalf(n=50, maxn=3_000, strict=True)
        return bracket_product

    @staticmethod
    def bracket_product_scipy_nquad(wave_function, wave_function_prime, operator=None):
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
        bracket_product = QuantumFactory.complex_quadrature(sc.integrate.nquad, integral_core_lambdify, [bound_phi, bound_r, bound_theta])[0]
        return bracket_product

    @staticmethod
    def bracket_product_scipy_tplquad(wave_function, wave_function_prime, operator=None):
        from sympy.utilities.lambdify import lambdastr
        r, theta, phi = sp.Symbol("r", real=True), sp.Symbol("theta", real=True), sp.Symbol("phi", real=True)
        jacobian = (r ** 2) * sp.sin(theta)
        integral_core = sp.conjugate(wave_function) * (operator if operator is not None else 1) * wave_function_prime

        # integral_core_expension = sp.expand(sp.FU["TR8"](jacobian * integral_core), func=True).simplify()
        integral_core_expension = sp.expand(jacobian * integral_core, func=True).simplify()

        # Process the integral with scipy
        integral_core_lambdify = sp.lambdify((theta, r, phi), integral_core_expension, modules="numpy")
        print(lambdastr((theta, r, phi), integral_core_expension))
        bracket_product = QuantumFactory.complex_quadrature(sc.integrate.tplquad, integral_core_lambdify,
                                                            0, 2*np.pi, lambda b_phi: 0, lambda b_phi: np.inf,
                                                            lambda b_theta, b_r: 0, lambda b_theta, b_r: np.pi)[0]
        # bracket_product = sc.integrate.tplquad(integral_core_lambdify,
        #                                        0, 2 * np.pi, lambda b_phi: 0, lambda b_phi: np.inf,
        #                                        lambda b_theta, b_r: 0, lambda b_theta, b_r: np.pi)[0]
        return bracket_product

    @staticmethod
    def complex_quadrature(integrate_func, func, *args, **kwargs):
        """

        :param integrate_func:
        :param func:
        :param args:
        :param kwargs:
        :return:
        """
        def real_func(x, y, z):
            return sc.real(func(x, y, z))

        def imag_func(x, y, z):
            return sc.imag(func(x, y, z))

        real_integral = integrate_func(real_func, *args, **kwargs)
        imag_integral = integrate_func(imag_func, *args, **kwargs)
        return real_integral[0] + 1j * imag_integral[0], real_integral[1:], imag_integral[1:]

    @staticmethod
    def bracket_product_mcint(wave_function, wave_function_prime, operator=None):
        r, theta, phi = sp.Symbol("r", real=True), sp.Symbol("theta", real=True), sp.Symbol("phi", real=True)
        jacobian = (r ** 2) * sp.sin(theta)
        integral_core = sp.conjugate(wave_function) * (operator if operator is not None else 1) * wave_function_prime

        # integral_core_expension = sp.expand(sp.FU["TR8"](jacobian * integral_core), func=True).simplify()
        integral_core_expension = sp.expand(jacobian * integral_core, func=True).simplify()

        integral_core_lambdify = sp.lambdify((theta, r, phi), integral_core_expension, modules="numpy")

        def integrand(x):
            theta, r, phi = x[0], x[1], x[2]
            return integral_core_lambdify(theta, r, phi)

        domainsize = (0, 1)
        result = QuantumFactory.monte_carlo_integration_sph(integrand, [[0, 1.15], [0, np.pi], [0, 2 * np.pi]], domainsize, 1_000_000)
        return result

    @staticmethod
    def monte_carlo_integration(integrand, bornes: list, domainsize=(0.0, 1.0), n_sample: int = int(1e6)):
        np.random.seed(1)
        # Sum elements and elements squared
        total = 0.0
        total_sq = 0.0
        count_in_curve = 0

        def sampler():
            while True:
                yield [np.random.uniform(b0, b1) for [b0, b1] in bornes]

        for x in itertools.islice(sampler(), n_sample):
            f = integrand(x)
            f_rn = np.random.uniform(domainsize[0], domainsize[1])
            total += f
            total_sq += (f ** 2)
            count_in_curve += 1 if 0 <= f_rn <= f else 0
        # Return answer
        # sample_mean = total / n_sample
        # sample_var = (total_sq - ((total / n_sample) ** 2) / n_sample) / (n_sample - 1.0)
        # return domainsize * sample_mean, domainsize * np.sqrt(sample_var / n_sample)
        v = np.prod([b1-b0 for [b0, b1] in bornes]) * (domainsize[1] - domainsize[0])
        print(100*(count_in_curve/n_sample), "%")
        return (count_in_curve/n_sample) * v

    @staticmethod
    def monte_carlo_integration_sph(integrand, bornes: list, domainsize=(0.0, 1.0), n_sample: int = int(1e6)):
        # np.random.seed(1)
        # Sum elements and elements squared

        # def sampler():
        #     while True:
        #         yield [np.random.uniform(b0, b1) for [b0, b1] in bornes]

        samples = np.array([[np.random.uniform(b0, b1) for [b0, b1] in bornes] for _ in range(n_sample)])
        # f_samples = np.array([np.random.uniform(domainsize[0], domainsize[1]) for _ in range(n_sample)])
        # for x in itertools.islice(sampler(), n_sample):
        # f_values = integrand(samples)
        # @numba.generated_jit(nopython=True)
        def loop():
            count: int = int(0)
            for i in prange(len(samples)):
                f = integrand(samples[i])
                # f = f_values[i]
                f_rn = np.random.uniform(domainsize[0], domainsize[1])
                # if (f > 0 and 0 <= f_samples[i] <= f) or (f < 0 and f <= f_samples[i] <= 0):
                if (f > 0 and 0 <= f_rn <= f) or (f < 0 and f <= f_rn <= 0):
                    count += 1
            return count

        count_in_curve = loop()
        # Return answer
        # sample_mean = total / n_sample
        # sample_var = (total_sq - ((total / n_sample) ** 2) / n_sample) / (n_sample - 1.0)
        # return domainsize * sample_mean, domainsize * np.sqrt(sample_var / n_sample)
        # v = ((4*np.pi*(bornes[0][1] - bornes[0][0])**3)/3) * (domainsize[1] - domainsize[0])
        # print(sp.integrate((4/3)*sp.pi*sp.Symbol('r')**3, (sp.Symbol('z'), domainsize[0], domainsize[1])))
        v = sp.integrate((4/3)*sp.pi*sp.Symbol('r')**3, (sp.Symbol('z'), domainsize[0], domainsize[1])).evalf(subs={'r': bornes[0][1] - bornes[0][0]})
        # print(100 * (count_in_curve / n_sample), "%")
        return (count_in_curve / n_sample) * v


if __name__ == '__main__':
    from Transitions import Transitions
    from QuantumState import QuantumState

    qs1 = QuantumState(n=2, ell=1, m_ell=1, s=0.5, m_s=0.5)
    qs2 = QuantumState(n=1, ell=0, m_ell=0, s=0.5, m_s=0.5)

    y1 = QuantumFactory.Y_ell_m_ell(3, 2)
    y2 = QuantumFactory.Y_ell_m_ell(2, 1)

    # def ylm(ell, m_ell):
    #     from scipy.special import sph_harm as ynm
    #     return ynm(ell, m_ell)

    # print(sp.conjugate(qs1.get_wave_fonction(z=const.Z_H, mu=const.mu_H).expand(func=True)), qs1.get_wave_fonction(z=const.Z_H, mu=const.mu_H).expand(func=True))
    #
    # print(QuantumFactory.bracket_product(qs1.get_wave_fonction(z=const.Z_H, mu=const.mu_H), qs2.get_wave_fonction(z=const.Z_H, mu=const.mu_H)))
    print(QuantumFactory.bracket_product(y1, y1))
    print(QuantumFactory.bracket_product(y1, y2))
    print(QuantumFactory.bracket_product(y2, y2))

