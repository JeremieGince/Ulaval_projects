from Transition import Transition
import Constantes as const
import numpy as np
import sympy as sp


class Transitions(list):
    def __init__(self, n: int = None, n_prime: int = None):
        super().__init__()
        assert (n is None and n_prime is None) or (isinstance(n, int) and isinstance(n, int))
        self.append_transitions_n_to_n(n, n_prime)
        self.spontanious_decay_mean = None

    def append(self, transition: Transition) -> None:
        super().append(transition)

    def append_transitions_n_to_n(self, n, n_prime):
        from QuantumState import QuantumState
        for trans in QuantumState.get_valid_transitions_n_to_n(n, n_prime):
            self.append(trans)

    def get_spontanious_decay_mean(self):
        if self.spontanious_decay_mean is not None:
            return self.spontanious_decay_mean
        rs_vector = np.array([trans.get_spontaniuous_decay_rate(z=const.ZH, mu=const.mu0) for trans in self])
        self.spontanious_decay_mean = np.mean(rs_vector)
        return self.spontanious_decay_mean

    def get_angular_frequency(self, n, n_prime, z=sp.Symbol("Z", real=True), mu=sp.Symbol('mu', real=True)):
        e = (- (z ** 2) * (const.alpha ** 2) * mu * (const.c ** 2))/(2 * (n ** 2))
        e_prime = (- (z ** 2) * (const.alpha ** 2) * mu * (const.c ** 2))/(2 * (n_prime ** 2))
        return (e - e_prime)/const.hbar

    def __repr__(self) -> str:
        this_repr = "[ "
        for trans in self:
            this_repr += f"{trans}, \n"
        this_repr += "]"
        return this_repr


if __name__ == '__main__':
    pass