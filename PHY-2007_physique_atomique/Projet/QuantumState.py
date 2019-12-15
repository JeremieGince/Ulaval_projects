import numpy as np
import Constants as const
import sympy as sp
from QuantumFactory import QuantumFactory


class QuantumState:

    def __init__(self, n: int, ell: int, m_ell: int, s: float, m_s: float, hydrogen: bool = False):
        """
        QuantumState constructor
        :param n: orbital number (int)
        :param ell: angular momentum (int)
        :param m_ell: quantum number m_ell (int)
        :param s: spin (float)
        :param m_s: quantum number m_s (float)
        :param hydrogen : if the current quantum state refer to a hydrogen atom (bool)
        """
        self._n: int = n
        self._ell: int = ell
        self._m_ell: int = m_ell
        self._s: float = s
        self._m_s: float = m_s
        self.hydrogen: bool = hydrogen
        self.check_invariants()

    def check_invariants(self) -> None:
        """
        Check every invariant for a quantum state
        :return: None
        """
        assert self._n >= 1
        assert 0 <= self._ell < self._n
        assert -self._ell <= self._m_ell <= self._ell
        assert self._s >= 0.0
        assert -self._s <= self._m_s <= self._s

    def getState(self) -> np.ndarray:
        """
        Getter of the state vector of state numbers in order (n, ell, m_ell, s, m_s)
        :return: [n, ell, m_ell, s, m_s] (numpy.ndarray)
        """
        return np.array([self._n, self._ell, self._m_ell, self._s, self._m_s])

    def get_n(self) -> int:
        return self._n

    def get_ell(self) -> int:
        return self._ell

    def get_m_ell(self) -> int:
        return self._m_ell

    def get_s(self) -> float:
        return self._s

    def get_m_s(self) -> float:
        return self._m_s

    def get_state_energy(self, z=sp.Symbol("Z", real=True), mu=sp.Symbol('mu', real=True)):
        """
        Get the energy of the current quantum state
        :param z: atomic number
        :param mu: reduced mass
        :return: the energy of the current quantum state (float if z and mu are float else sympy object)
        """
        return QuantumFactory.get_state_energy_unperturbed(self._n, z, mu)

    def get_valid_transitions_state_to_n(self, other_n: int) -> list:
        """
        Get all of the valid transition of the current quantum state to another in the orbital other_n
        :param other_n: (int)
        :return: list of QuantumState
        """
        import warnings
        warnings.warn("Warning! This method seems to be not valid", DeprecationWarning)
        from Transition import Transition
        valid_transitions = list()
        next_states = set()
        for key, possibilities in Transition.TRANSITIONS_RULES.items():
            for num in possibilities:
                for key_prime, possibilities_prime in Transition.TRANSITIONS_RULES.items():
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

    def __repr__(self) -> str:
        """
        show a representation of the current quantum state
        :return: string representation of self (str)
        """
        this_repr = f"(n: {self._n}, " \
                    f"ell: {self._ell}, " \
                    f"m_ell: {self._m_ell}, " \
                    f"s: {self._s}, " \
                    f"m_s: {self._m_s})"
        return this_repr

    def repr_without_spin(self) -> str:
        """
        show a representation of the current quantum state without s and m_s
        :return: string representation of self without orbital spin (str)
        """
        this_repr = f"(n: {self._n}, " \
                    f"ell: {self._ell}, " \
                    f"m_ell: {self._m_ell})"
        return this_repr

    def get_wave_function(self, z=sp.Symbol("Z", real=True), mu=sp.Symbol('mu', real=True)):
        """
        Get the wave function of the current quantum state as a sympy object
        :param z: atomic number
        :param mu: reduced mass (float)
        :return: the wave function (sympy object)
        """
        if self.hydrogen:
            return QuantumFactory.get_hydrogen_wave_function(self._n, self._ell, self._m_ell)

        u = QuantumFactory.u_n_ell(n=self._n, ell=self._ell, z=z, mu=mu)
        y = QuantumFactory.Y_ell_m_ell(self._ell, self._m_ell)
        return u*y

    def decay_number(self, T: float, z: int = const.Z_H, mu: float = const.mu_H):
        """
        Return the decay number of the current QuantumState.
        :param T: Current temperature (float)
        :param z: atomic number (int)
        :param mu: reduced mass (float)
        :return: a sympy expression of the decay number (sympy object)
        """
        alpha = sp.Symbol('alpha')  # proportional function
        return alpha*QuantumFactory.get_g_n(self._n)*sp.exp(-self.get_state_energy(z, mu)/(const.k_B*T))


if __name__ == '__main__':
    from Transition import Transition
    quantum_state = QuantumState(n=1, ell=0, m_ell=0, s=1 / 2, m_s=1 / 2)
    print(f"psi_{quantum_state} = {quantum_state.get_wave_function()}")
    print(f"E_{quantum_state} = {quantum_state.get_state_energy()}")
    print(Transition.possible(quantum_state, QuantumState(2, 0, 0, 1 / 2, 1 / 2)))
    print(Transition.possible(quantum_state, QuantumState(2, 1, 0, 1/2, 1 / 2)))

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

    print(QuantumFactory.get_valid_ell_with_n(3))

    print('-' * 175)

    for valid_transition in QuantumFactory.get_valid_transitions_n_to_n(1, 2):
        print(valid_transition)