import numpy as np
import Constantes as const
import sympy as sp


class QuantumState:
    transition_rules = {"Delta ell": [-1, 1],
                        "Delta m_ell": [-1, 0, 1],
                        "Delta s": [0],
                        "Delta m_s": [0]}

    def __init__(self, n, ell, m_ell, s, m_s):
        self._n = n
        self._ell = ell
        self._m_ell = m_ell
        self._s = s
        self._m_s = m_s
        self.check_invariants()

    def check_invariants(self, assertion: bool = True) -> bool:
        check = True
        if not (self._n >= 1):
            check = False
        if not (0 <= self._ell < self._n):
            check = False
        if not (-self._ell <= self._m_ell <= self._ell):
            check = False
        if not (self._s >= 0.0):
            check = False
        if not (-self._s <= self._m_s <= self._s):
            check = False

        if assertion and not check:
            raise AssertionError(f"{self}")
        return check

    def getState(self):
        return np.array([self._n, self._ell, self._m_ell, self._s, self._m_s])

    def get_n(self):
        return self._n

    def get_ell(self):
        return self._ell

    def get_m_ell(self):
        return self._m_ell

    def get_s(self):
        return self._s

    def get_m_s(self):
        return self._m_s

    def get_state_energy(self, z=1, mu=const.mu0):
        numerator = - (z**2)*(const.alpha**2)*mu*(const.c**2)
        denumerator = 2*(self._n**2)
        return numerator/denumerator

    def transition(self, other):
        from Transition import Transition
        assert isinstance(other, QuantumState), "other must be an instance of QuantumState"
        assert Transition.possible(self, other), "The transition must be valid. " \
                                                 "Make sur QuantumState.able_to_translate(other) return True."
        self._n = other._n
        self._ell = other._ell
        self._m_ell = other._m_ell
        self._s = other._s
        self._m_s = other._m_s

        "return transition_energy"

    def get_valid_transitions_state_to_n(self, other_n: int) -> list:
        from Transition import Transition
        valid_transitions = list()
        next_states = set()
        for key, possibilities in self.transition_rules.items():
            for num in possibilities:
                for key_prime, possibilities_prime in self.transition_rules.items():
                    if key == key_prime:
                        continue
                    for num_prime in possibilities_prime:
                        (_, other_ell, other_m_ell, other_s, other_m_s) = tuple(self.getState())
                        if key == "Delta ell":
                            other_ell += num
                        elif key == "Delta m_ell":
                            other_m_ell += num
                        elif key == "Delta s":
                            other_s += num
                        elif key == "Delta m_s":
                            other_m_s += num

                        if key_prime == "Delta ell":
                            other_ell += num_prime
                        elif key_prime == "Delta m_ell":
                            other_m_ell += num_prime
                        elif key_prime == "Delta s":
                            other_s += num_prime
                        elif key_prime == "Delta m_s":
                            other_m_s += num_prime
                        try:
                            next_state = QuantumState(other_n, other_ell, other_m_ell, other_s, other_m_s)
                        except AssertionError:
                            continue
                        if Transition.possible(self, next_state):
                            if str(next_state) not in next_states:
                                valid_transitions.append(next_state)
                            next_states.add(str(next_state))
                        else:
                            continue

        return valid_transitions

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
    def get_valid_transitions_n_to_n(n, n_prime) -> list:
        valid_transitions = list()
        for ell in QuantumState.get_valid_ell_with_n(n):
            for m_ell in QuantumState.get_valid_m_ell_with_ell(ell):
                for s in [1/2]:
                    for m_s in QuantumState.get_valid_m_s_with_s(s):
                        init_quantum_state = QuantumState(n, ell, m_ell, s, m_s)
                        for end_quantum_state in QuantumState.get_valid_transitions_state_to_n(init_quantum_state, n_prime):
                            valid_transitions.append(Transition(init_quantum_state, end_quantum_state))
        return valid_transitions

    def __repr__(self):
        this_repr = f"(n: {self._n}, " \
                    f"ell: {self._ell}, " \
                    f"m_ell: {self._m_ell}, " \
                    f"s: {self._s}, " \
                    f"m_s: {self._m_s})"
        return this_repr

    def get_wave_fonction(self, z=1, mu=const.mu0):
        theta = sp.Symbol("theta")
        phi = sp.Symbol("phi")
        r = sp.Symbol("r")
        y_ell_m_ell = sp.Ynm(self._ell, self._m_ell, theta, phi)

        coeff = np.sqrt((((2*z*const.alpha*mu*const.c)
                          /(self._n*const.hbar))**3)*(np.math.factorial(self._n-self._ell-1)
                                                      /(2*self._n*np.math.factorial(self._n-self._ell))))
        xi_n = (2*z*const.alpha*mu*const.c*r)/(self._n*const.hbar)
        u_n_ell = coeff*sp.exp(-xi_n/2)*(xi_n**self._ell)*sp.assoc_laguerre(self._n-self._ell-1,
                                                                            2*self._ell+1, xi_n)

        psi_n_ell_m_ell = u_n_ell*y_ell_m_ell
        return psi_n_ell_m_ell


if __name__ == '__main__':
    from Transition import Transition
    quantum_state = QuantumState(n=1, ell=0, m_ell=0, s=1 / 2, m_s=1 / 2)
    print(quantum_state.getState())
    print(Transition.possible(quantum_state, QuantumState(2, 0, 0, 1 / 2, 1 / 2)))
    print(Transition.possible(quantum_state, QuantumState(2, 1, 1/2, 1/2, 1 / 2)))

    valid_transitions_test_1_to_2 = [QuantumState(2, 1, -1, 1/2, 1/2),
                                     QuantumState(2, 1, 0, 1/2, 1/2),
                                     QuantumState(2, 1, 1, 1/2, 1/2)]

    for state in QuantumState(n=1, ell=0, m_ell=0, s=1/2, m_s=1/2).get_valid_transitions_state_to_n(other_n=2):
        print(state.getState())

    print('-'*175)

    valid_transitions_test_3_to_1 = [QuantumState(1, 0, 0, 1/2, 1/2)]

    for state in QuantumState(n=3, ell=1, m_ell=0, s=1/2, m_s=1/2).get_valid_transitions_state_to_n(other_n=1):
        print(state.getState())

    print('-'*175)

    print(QuantumState.get_valid_ell_with_n(3))

    print('-' * 175)

    for valid_transition in QuantumState.get_valid_transitions_n_to_n(1, 2):
        print(valid_transition)