from QuantumState import QuantumState
import Constants as const
import sympy as sp
import numpy as np
import numba
from QuantumFactory import QuantumFactory
import os


class Transition:
    TRANSITIONS_RULES: dict = {"Delta ell": [-1, 1],
                               "Delta m_ell": [-1, 0, 1],
                               "Delta s": [0],
                               "Delta m_s": [0]}

    TRANSITIONS_RULES_SPIN: dict = {"Delta ell": [-1, 1],
                                    "Delta m_ell": [-1, 0, 1],
                                    "Delta j": [-1, 0, 1],
                                    "Delta m_j": [-1, 0, 1]}

    n_ell_m_ell_state_to_rs: dict = dict()

    transition_count: int = int(0)

    static_save_file = os.getcwd() + "/Transition_static_values.npy"

    def __init__(self, initial_quantum_state: QuantumState, ending_quantum_state: QuantumState):
        """
        Transition constructor
        :param initial_quantum_state: initial quantum state (QuantumState)
        :param ending_quantum_state: final quantum state (QuantumState)
        """
        self._initial_quantum_state = initial_quantum_state
        self._ending_quantum_state = ending_quantum_state
        self.check_invariant()
        self.spontaneous_decay_rate = None
        Transition.transition_count += 1
        self.load_static_values()

    def __del__(self):
        Transition.save_static_values()
        Transition.transition_count -= 1

    @staticmethod
    def save_static_values() -> None:
        if Transition.transition_count >= 1:
            np.save(Transition.static_save_file, Transition.n_ell_m_ell_state_to_rs)

    @staticmethod
    def load_static_values() -> None:
        if Transition.transition_count >= 1 and os.path.exists(Transition.static_save_file):
            Transition.n_ell_m_ell_state_to_rs = np.load(Transition.static_save_file, allow_pickle=True).item()

    @staticmethod
    def clear_static_values() -> None:
        os.remove(Transition.static_save_file)

    def check_invariant(self) -> None:
        """
        Check every invariant for a Transition
        :return: None
        """
        assert Transition.possible(self._initial_quantum_state, self._ending_quantum_state)

    def get_n_to_n_prime_couple(self) -> tuple:
        """
        Gives the tuple (initial_quantum_state.n, ending_quantum_state.n)
        :return: (tuple)
        """
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

    def get_spontaneous_decay_rate(self, z=sp.Symbol('Z', real=True), mu=sp.Symbol('mu', real=True), algo="auto"):
        """
        Get the spontaneous decay rate of the transition
        :param z: atomic number (int)
        :param mu: reduced mass (float)
        :return: the spontaneous decay rate (float)
        """
        # check if spontaneous_decay_rate is already calculated
        if self.spontaneous_decay_rate is not None:
            return self.spontaneous_decay_rate
        elif self.repr_without_spin() in Transition.n_ell_m_ell_state_to_rs.keys():
            self.spontaneous_decay_rate = Transition.n_ell_m_ell_state_to_rs[self.repr_without_spin()]
            return self.spontaneous_decay_rate

        r, theta, phi = sp.Symbol("r", real=True, positive=True), sp.Symbol("theta", real=True),\
                        sp.Symbol("phi", real=True)
        coeff = (4*const.alpha*(self.get_delta_energy(z, mu)**3))/(3*(const.hbar**3)*(const.c**2))
        psi = self._initial_quantum_state.get_wave_function(z, mu)
        psi_prime = self._ending_quantum_state.get_wave_function(z, mu)

        if algo == "auto":
            if self._initial_quantum_state.hydrogen and self._ending_quantum_state.hydrogen:
                algo = "sympy_ns"
            else:
                algo = "scipy_nquad"

        # \Vec{r} = x + y + z
        r_operator = r * sp.cos(theta) * (sp.cos(phi) + sp.sin(phi) + 1)

        # Process the integral with algo
        bracket_product = QuantumFactory.bracket_product(psi, psi_prime, r_operator, algo)

        bracket_product_norm_square = sp.Mul(sp.conjugate(bracket_product), bracket_product).evalf()
        # print(bracket_product_norm_square)

        self.spontaneous_decay_rate = sp.Float(coeff * bracket_product_norm_square)

        # updating Transition static attribute
        Transition.n_ell_m_ell_state_to_rs[self.repr_without_spin()] = self.spontaneous_decay_rate
        return self.spontaneous_decay_rate

    def get_delta_energy(self, z=sp.Symbol('Z', real=True), mu=sp.Symbol('mu', real=True)):
        """
        Getter of the transition energy without any perturbation
        :param z: atomic number (float)
        :param mu: reduced mass (float)
        :return: transition energy (float) or transition energy (sympy object)
        """
        return self._initial_quantum_state.get_state_energy(z, mu) - self._ending_quantum_state.get_state_energy(z, mu)

    def get_angular_frequency(self, z=sp.Symbol('Z', real=True), mu=sp.Symbol('mu', real=True)):
        """
        Getter of the transition angular frequency without any perturbation
        :param z: atomic number (float)
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

    qs1 = QuantumState(n=2, ell=1, m_ell=0, s=0.5, m_s=0.5, hydrogen=True)
    qs2 = QuantumState(n=1, ell=0, m_ell=0, s=0.5, m_s=0.5, hydrogen=True)

    print(f"psi_1{qs1} = {qs1.get_wave_function()}")
    print(f"psi_2{qs2} = {qs2.get_wave_function()}")

    rs_mean_norm_coeff = ((const.alpha ** 5) * (const.c ** 2) * const.mu_H) / const.hbar
    omega_normalized_coeff = ((const.alpha ** 2) * const.mu_H * (const.c ** 2)) / (2 * const.hbar)

    trans = Transition(qs1, qs2)

    print(Transition.n_ell_m_ell_state_to_rs)

    print('-' * 175)
    rs_normalized = trans.get_spontaneous_decay_rate(z=const.Z_H, mu=const.mu_H, algo="auto") / rs_mean_norm_coeff
    reel_rs_normalized = 0.002746686
    print(f"Transition: {trans}")
    print(f"R^s sympy = {rs_normalized}")
    print(f"reel R^s = {reel_rs_normalized},"
          f" deviation: {100*(np.abs(rs_normalized-reel_rs_normalized))} %")
    print('-'*175+f'\n elapse time: {time.time()-start_time}')

    start_time = time.time()

    print(Transition.transition_count)
    del(trans)
    print(Transition.transition_count)


