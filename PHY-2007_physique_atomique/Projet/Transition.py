from QuantumState import QuantumState
import Constantes as const
import sympy as sp
import numpy as np


class Transition:
    transition_rules = {"Delta ell": [-1, 1],
                        "Delta m_ell": [-1, 0, 1],
                        "Delta s": [0],
                        "Delta m_s": [0]}

    def __init__(self, initial_quantum_state: QuantumState, ending_quantum_state: QuantumState):
        self._initial_quantum_state = initial_quantum_state
        self._ending_quantum_state = ending_quantum_state
        self.check_invariant()

    def check_invariant(self):
        assert Transition.possible(self._initial_quantum_state, self._ending_quantum_state)

    def __repr__(self):
        this_repr = f"({self._initial_quantum_state} -> {self._ending_quantum_state})"
        return this_repr

    def get_spontaniuous_decay_rate(self):
        r, theta, phi = sp.Symbol("r", real=True), sp.Symbol("theta", real=True), sp.Symbol("phi", real=True)
        coeff = (4*const.alpha*(self._initial_quantum_state.get_state_energy()
                                - self._ending_quantum_state.get_state_energy())**3)\
                /(3*(const.hbar**3)*(const.c**2))
        psi = self._initial_quantum_state.get_wave_fonction()
        psi_prime = self._ending_quantum_state.get_wave_fonction()

        bracket_product = sp.integrate((r**2)*sp.sin(theta)*sp.conjugate(psi)*r*psi_prime,
                                       (phi, 0, 2*sp.pi), (theta, 0, sp.pi), (r, 0, sp.oo)).evalf()
        # print(bracket_product)
        bracket_product_norm_square = (sp.conjugate(bracket_product)*bracket_product).evalf()
        return sp.Float(coeff*bracket_product_norm_square)

    @staticmethod
    def possible(initial_quantum_state: QuantumState, ending_quantum_state: QuantumState):
        able = True

        if initial_quantum_state.get_n() == ending_quantum_state.get_n():
            able = False

        if (initial_quantum_state.get_ell() - ending_quantum_state.get_ell()) not in Transition.transition_rules[
            "Delta ell"]:
            able = False

        if (initial_quantum_state.get_m_ell() - ending_quantum_state.get_m_ell()) not in Transition.transition_rules[
            "Delta m_ell"]:
            able = False

        if (initial_quantum_state.get_s() - ending_quantum_state.get_s()) not in Transition.transition_rules["Delta s"]:
            able = False

        if (initial_quantum_state.get_m_s() - ending_quantum_state.get_m_s()) not in Transition.transition_rules[
            "Delta m_s"]:
            able = False

        return able


if __name__ == '__main__':
    qs1 = QuantumState(n=1, ell=0, m_ell=0, s=0.5, m_s=-0.5)
    qs2 = QuantumState(n=2, ell=1, m_ell=-1, s=0.5, m_s=-0.5)

    print(f"psi_1{qs1} = {qs1.get_wave_fonction()}")
    print(f"psi_2{qs2} = {qs2.get_wave_fonction()}")

    trans = Transition(qs1, qs2)
    print(f"Transition: {trans}")
    print(f"R^s = {trans.get_spontaniuous_decay_rate()}")