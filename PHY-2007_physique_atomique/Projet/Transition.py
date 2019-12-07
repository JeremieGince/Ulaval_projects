from QuantumState import QuantumState
import Constantes as const
import sympy as sp
import numpy as np
import numba
from numba import cuda


class Transition:
    transition_rules = {"Delta ell": [-1, 1],
                        "Delta m_ell": [-1, 0, 1],
                        "Delta s": [0],
                        "Delta m_s": [0]}

    def __init__(self, initial_quantum_state: QuantumState, ending_quantum_state: QuantumState):
        self._initial_quantum_state = initial_quantum_state
        self._ending_quantum_state = ending_quantum_state
        self.check_invariant()
        self.spontaniuous_decay_rate = None

    def check_invariant(self):
        assert Transition.possible(self._initial_quantum_state, self._ending_quantum_state)

    def __repr__(self):
        this_repr = f"({self._initial_quantum_state} -> {self._ending_quantum_state})"
        return this_repr

    @numba.jit(parallel=True)
    def get_spontaniuous_decay_rate(self, z=sp.Symbol('Z', real=True), mu=sp.Symbol('mu', real=True)):
        if self.spontaniuous_decay_rate is not None:
            return self.spontaniuous_decay_rate

        r, theta, phi = sp.Symbol("r", real=True), sp.Symbol("theta", real=True), sp.Symbol("phi", real=True)
        coeff = (4*const.alpha*(self._initial_quantum_state.get_state_energy(z, mu)
                                - self._ending_quantum_state.get_state_energy(z, mu))**3)/(3*(const.hbar**3)*(const.c**2))
        psi = self._initial_quantum_state.get_wave_fonction(z, mu)
        psi_prime = self._ending_quantum_state.get_wave_fonction(z, mu)

        integral_core = (r**3)*sp.FU['TR8'](sp.sin(theta)*sp.cos(theta))*sp.conjugate(psi)*psi_prime

        print(integral_core)

        normalized_coeff = 1

        bracket_product = sp.Integral(sp.FU['TR0'](integral_core.simplify()),
                                      (phi, 0, 2*sp.pi), (theta, 0, sp.pi), (r, 0, sp.oo)).doit()
        # bracket_product = sp.Integral(sp.FU['TR0'](integral_core.simplify()), (phi, 0, 2 * sp.pi)).as_sum().n()
        # bracket_product = sp.Integral(bracket_product, (theta, 0, sp.pi)).as_sum()
        # bracket_product = sp.Integral(bracket_product, (r, 0, sp.oo)).as_sum()

        print((bracket_product/normalized_coeff))
        bracket_product = sp.FU['TR0'](bracket_product/normalized_coeff).evalf(20)
        print(bracket_product)
        # bracket_product_norm_square = (sp.conjugate(bracket_product)*bracket_product).evalf()
        bracket_product_norm_square = sp.Mul(sp.conjugate(bracket_product), bracket_product).evalf()
        print(bracket_product_norm_square)
        self.spontaniuous_decay_rate = sp.Float(coeff*bracket_product_norm_square)*normalized_coeff
        return self.get_spontaniuous_decay_rate

    def get_angular_frequency(self):
        delta_e = self._initial_quantum_state.get_state_energy() - self._ending_quantum_state.get_state_energy()
        return delta_e/const.hbar

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
    import time
    start_time = time.time()
    print(cuda.gpus)
    qs1 = QuantumState(n=1, ell=0, m_ell=0, s=0.5, m_s=-0.5)
    qs2 = QuantumState(n=2, ell=1, m_ell=-1, s=0.5, m_s=-0.5)

    print(f"psi_1{qs1} = {qs1.get_wave_fonction()}")
    print(f"psi_2{qs2} = {qs2.get_wave_fonction()}")

    trans = Transition(qs1, qs2)
    print(f"Transition: {trans}")
    print(f"R^s = {trans.get_spontaniuous_decay_rate(z=const.ZH, mu=const.mu0)}")
    print('-'*175+f'\n elapse time: {time.time()-start_time}')
