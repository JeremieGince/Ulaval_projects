import numpy as np


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
        if not (self._s >= 0):
            check = False
        if not (-self._s <= self._m_s <= self._s):
            check = False

        if assertion and not check:
            raise AssertionError()
        return check

    def getState(self):
        return np.array([self._n, self._ell, self._m_ell, self._s, self._m_s])

    def able_to_translate(self, other) -> bool:
        assert isinstance(other, QuantumState), "other must be an instance of QuantumState"
        able = True

        if self._n == other._n:
            able = False

        if (self._ell - other._ell) not in self.transition_rules["Delta ell"]:
            able = False

        if (self._m_ell - other._m_ell) not in self.transition_rules["Delta m_ell"]:
            able = False

        if (self._s - other._s) not in self.transition_rules["Delta s"]:
            able = False

        if (self._m_s - other._m_s) not in self.transition_rules["Delta m_s"]:
            able = False

        return able

    def transition(self, other):
        assert isinstance(other, QuantumState), "other must be an instance of QuantumState"
        assert self.able_to_translate(other), "The transition must be valid. " \
                                              "Make sur QuantumState.able_to_translate(other) return True."
        self._n = other._n
        self._ell = other._ell
        self._m_ell = other._m_ell
        self._s = other._s
        self._m_s = other._m_s

        "return transition_energy"

    def get_valid_transitions_state_to_n(self, other_n: int) -> list:
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
                        if self.able_to_translate(next_state):
                            if str(next_state.getState()) not in next_states:
                                valid_transitions.append(next_state)
                            next_states.add(str(next_state.getState()))
                        else:
                            continue

        return valid_transitions

    @staticmethod
    def get_valid_transitions_n_to_n(self, n, n_prime):
        pass

    def __repr__(self):
        this_repr = f"n: {self._n} \n" \
                    f"ell: {self._ell} \n" \
                    f"m_ell: {self._m_ell} \n" \
                    f"s: {self._s} \n" \
                    f"m_s: {self._m_s}"
        return this_repr


if __name__ == '__main__':
    quantum_state = QuantumState(n=1, ell=0, m_ell=0, s=1 / 2, m_s=1 / 2)
    print(quantum_state.getState())
    print(quantum_state.able_to_translate(QuantumState(2, 0, 1 / 2, 1 / 2, 1 / 2)))
    print(quantum_state.able_to_translate(QuantumState(2, 1, 1 / 2, 1 / 2, 1 / 2)))

    valid_transitions_test_1_to_2 = [QuantumState(2, 1, -1, 1/2, 1/2),
                                     QuantumState(2, 1, 0, 1/2, 1/2),
                                     QuantumState(2, 1, 1, 1/2, 1/2)]

    for state in QuantumState(n=1, ell=0, m_ell=0, s=1/2, m_s=1/2).get_valid_transitions_state_to_n(other_n=2):
        print(state.getState())

    print('-'*175)

    valid_transitions_test_3_to_1 = [QuantumState(1, 0, 0, 1/2, 1/2)]

    for state in QuantumState(n=3, ell=1, m_ell=0, s=1/2, m_s=1/2).get_valid_transitions_state_to_n(other_n=1):
        print(state.getState())
