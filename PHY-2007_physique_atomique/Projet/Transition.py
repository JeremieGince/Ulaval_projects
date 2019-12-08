from QuantumState import QuantumState
import Constantes as const
import sympy as sp
import numpy as np
import numba
from numba import cuda
from sympy.utilities.lambdify import lambdastr
import mpmath


class Transition:
    TRANSITIONS_RULES = {"Delta ell": [-1, 1],
                         "Delta m_ell": [-1, 0, 1],
                         "Delta s": [0],
                         "Delta m_s": [0]}

    TRANSITIONS_RULES_SPIN = {"Delta ell": [-1, 1],
                              "Delta m_ell": [-1, 0, 1],
                              "Delta j": [-1, 0, 1],
                              "Delta m_j": [-1, 0, 1]}

    n_ell_m_ell_state_to_rs = dict()

    def __init__(self, initial_quantum_state: QuantumState, ending_quantum_state: QuantumState):
        """
        Transition constructor
        :param initial_quantum_state: initial quantum state (QuantumState)
        :param ending_quantum_state: final quantum state (QuantumState)
        """
        self._initial_quantum_state = initial_quantum_state
        self._ending_quantum_state = ending_quantum_state
        self.check_invariant()
        self.spontanious_decay_rate = None

    def check_invariant(self) -> None:
        """
        Check every invariant for a Transition
        :return: None
        """
        assert Transition.possible(self._initial_quantum_state, self._ending_quantum_state)

    def get_n_to_n_prime_couple(self) -> tuple:
        return self._initial_quantum_state.get_n(), self._ending_quantum_state.get_n()

    def __repr__(self) -> str:
        """
        show a representation of the current Transition
        :return: string representation of self (str)
        """
        this_repr = f"({self._initial_quantum_state} -> {self._ending_quantum_state})"
        return this_repr

    def repr_without_spin(self) -> str:
        """
        show a representation of the current Transition without s and m_s
        :return: string representation of self without orbital spin (str)
        """
        this_repr = f"({self._initial_quantum_state.repr_without_spin()} " \
                    f"-> {self._ending_quantum_state.repr_without_spin()})"
        return this_repr

    # @numba.jit
    def get_spontanious_decay_rate(self, z=sp.Symbol('Z', real=True), mu=sp.Symbol('mu', real=True)):
        """
        Get the spontanious decay rate of the transition
        :param z: (int)
        :param mu: reduced mass (float)
        :return: the spontanious decay rate (float)
        """
        # print(self.repr_without_spin())

        # check if spontanious_decay_rate is already calculated
        if self.spontanious_decay_rate is not None:
            # print("we already calculate it")
            return self.spontanious_decay_rate
        elif self.repr_without_spin() in Transition.n_ell_m_ell_state_to_rs.keys():
            self.spontanious_decay_rate = Transition.n_ell_m_ell_state_to_rs[self.repr_without_spin()]
            # print(f"n_ell_m_ell_state_to_rs: {Transition.n_ell_m_ell_state_to_rs}")
            return self.spontanious_decay_rate

        r, theta, phi = sp.Symbol("r", real=True), sp.Symbol("theta", real=True), sp.Symbol("phi", real=True)
        coeff = (4*const.alpha*(self.get_delta_energy(z, mu)**3))/(3*(const.hbar**3)*(const.c**2))
        psi = sp.FU['TR8'](self._initial_quantum_state.get_wave_fonction(z, mu))
        psi_prime = sp.FU['TR8'](self._ending_quantum_state.get_wave_fonction(z, mu))

        integral_core = (r**3)*sp.FU['TR8'](sp.sin(theta)*sp.cos(theta))*sp.conjugate(psi)*psi_prime
        # print(f"\n integral_core: {integral_core}")

        # print(sp.lambdify((r, theta, phi), integral_core))
        # print(lambdastr((r, theta, phi), integral_core))
        # print(sp.lambdify((r, theta, phi), integral_core)(0.1, 0.2, 0.3))
        # print(mpmath.quad(sp.lambdify((r, theta, phi), integral_core), [phi, 0, 2*mpmath.pi], [r, 0, mpmath.inf], [theta, 0, mpmath.pi]))
        # print('-'*75)
        # raise NotImplemented

        # creation of the Integral object and first try to resolve it
        bracket_product = sp.Integral(sp.FU['TR0'](integral_core.simplify()),
                                      (phi, 0, 2*sp.pi), (r, 0, mpmath.inf), (theta, 0, sp.pi)).doit()
        # print(f"\n Integral bracket_product: {bracket_product}")

        # simplify the result of the first try and evaluation of the integral, last attempt
        bracket_product = sp.FU['TR0'](bracket_product).evalf()
        # print(f"\n bracket_product: {bracket_product}")

        bracket_product_norm_square = sp.Mul(sp.conjugate(bracket_product), bracket_product).evalf()
        # print(bracket_product_norm_square)

        self.spontanious_decay_rate = sp.Float(coeff * bracket_product_norm_square)

        # updating Transition static attribute
        Transition.n_ell_m_ell_state_to_rs[self.repr_without_spin()] = self.spontanious_decay_rate
        return self.spontanious_decay_rate

    def get_delta_energy(self, z=sp.Symbol('Z', real=True), mu=sp.Symbol('mu', real=True)):
        """
        Getter of the transition energy without any pertubation
        :param z:
        :param mu: reduced mass (float)
        :return: transition energy (float) or transition energy (sympy object)
        """
        return self._initial_quantum_state.get_state_energy(z, mu) - self._ending_quantum_state.get_state_energy(z, mu)

    def get_angular_frequency(self, z=sp.Symbol('Z', real=True), mu=sp.Symbol('mu', real=True)):
        """
        Getter of the transition angular frequency without any pertubation
        :param z:
        :param mu: reduced mass (float)
        :return: angular frequency (float) or angular frequency (sympy object)
        """
        return self.get_delta_energy(z, mu)/const.hbar

    @staticmethod
    def possible(initial_quantum_state: QuantumState, ending_quantum_state: QuantumState) -> bool:
        """
        Check if the transition is possible with every transition rule selection
        :param initial_quantum_state: initial quantum state (QuantumState)
        :param ending_quantum_state: final quantum state (QuantumState)
        :return: a boolean representation of the capability of the transition (bool)
        """
        able = True

        if initial_quantum_state.get_n() == ending_quantum_state.get_n():
            able = False

        if (initial_quantum_state.get_ell() - ending_quantum_state.get_ell()) \
                not in Transition.TRANSITIONS_RULES["Delta ell"]:
            able = False

        if (initial_quantum_state.get_m_ell() - ending_quantum_state.get_m_ell()) \
                not in Transition.TRANSITIONS_RULES["Delta m_ell"]:
            able = False

        if (initial_quantum_state.get_s() - ending_quantum_state.get_s()) \
                not in Transition.TRANSITIONS_RULES["Delta s"]:
            able = False

        if (initial_quantum_state.get_m_s() - ending_quantum_state.get_m_s()) \
                not in Transition.TRANSITIONS_RULES["Delta m_s"]:
            able = False

        return able


if __name__ == '__main__':
    import time
    start_time = time.time()
    print(cuda.gpus)
    qs1 = QuantumState(n=2, ell=1, m_ell=0, s=0.5, m_s=0.5)
    qs2 = QuantumState(n=1, ell=0, m_ell=0, s=0.5, m_s=0.5)

    print(f"psi_1{qs1} = {qs1.get_wave_fonction()}")
    print(f"psi_2{qs2} = {qs2.get_wave_fonction()}")

    rs_mean_normalized_coeff = ((const.alpha ** 5) * const.mu_H * (const.c ** 2)) / const.hbar
    omega_normalized_coeff = ((const.alpha ** 2) * const.mu_H * (const.c ** 2)) / (2 * const.hbar)

    trans = Transition(qs1, qs2)
    rs_normalized = trans.get_spontanious_decay_rate(z=const.Z_H, mu=const.mu_H) / rs_mean_normalized_coeff
    reel_rs_normalized = 0.002746686
    print(f"Transition: {trans}")
    print(f"R^s = {rs_normalized}")
    print(f"reel R^s = {reel_rs_normalized},"
          f" deviation: {100*(np.abs(rs_normalized-reel_rs_normalized)/reel_rs_normalized)} %")
    print('-'*175+f'\n elapse time: {time.time()-start_time}')
