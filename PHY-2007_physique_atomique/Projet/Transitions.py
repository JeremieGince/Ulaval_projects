from Transition import Transition
import Constantes as const
import numpy as np
import sympy as sp
import tqdm


class Transitions(list):
    def __init__(self, n: int = None, n_prime: int = None, hydrogen: bool = False):
        """
        Transitions constructor. Transitions is a container of Transition object
         that can calculate some stats on these transitions
        :param n: initial orbital number n (int)
        :param n_prime: final orbital number n (int)
        :param hydrogen : if the current container of quantum states refer to hydrogen atoms (bool)
        """
        super().__init__()
        assert (n is None and n_prime is None) or (isinstance(n, int) and isinstance(n, int)),\
            "params n and n_prime must be integer or None"
        if isinstance(n, int) and isinstance(n, int):
            self.append_transitions_n_to_n(n, n_prime, hydrogen=hydrogen)
        self.spontanious_decay_mean: float = None

    def append(self, transition: Transition) -> None:
        """
        Add a transition in the current container
        :param transition: Transition to be added (Transition)
        :return: None
        """
        super().append(transition)

    def append_transitions_n_to_n(self, n, n_prime, hydrogen: bool = False) -> None:
        """
        Add all the possible transition between orbital number n and n_prime
        :param n: initial orbital number n (int)
        :param n_prime: final orbital number n (int)
        :param hydrogen: if we want to cast quantum state in quantum hydrogen state (bool)
        :return: None
        """
        from QuantumFactory import QuantumFactory
        for trans in QuantumFactory.get_valid_transitions_n_to_n_prime(n, n_prime, hydrogen=hydrogen):
            self.append(trans)

    def get_spontanious_decay_mean(self, z=const.Z_H, mu=const.mu_H) -> float:
        """
        Get mean of the spontanious decay rate of transitions in the current container
        :param z: (int)
        :param mu: redeced mass (float)
        :return: mean spontanious decay rate (float)
        """
        if self.spontanious_decay_mean is not None:
            return self.spontanious_decay_mean
        rs_vector: np.ndarray = np.array([trans.get_spontanious_decay_rate(z=z, mu=mu) for trans in tqdm.tqdm(self)])
        self.spontanious_decay_mean: float = np.float(np.mean(rs_vector))
        return self.spontanious_decay_mean

    @staticmethod
    def get_angular_frequency(n, n_prime, z=sp.Symbol("Z", real=True), mu=sp.Symbol('mu', real=True)) -> float:
        """
        Get the angular frequency between two states using the unperturbeted energy.
        :param n: initial orbital number n (int)
        :param n_prime: final orbital number n (int)
        :param z: (int)
        :param mu: (float)
        :return: angular frequency (float)
        """
        import warnings
        warnings.warn("Warning! This method seems to be replicated and not efficient", DeprecationWarning)
        e = (- (z ** 2) * (const.alpha ** 2) * mu * (const.c ** 2))/(2 * (n ** 2))
        e_prime = (- (z ** 2) * (const.alpha ** 2) * mu * (const.c ** 2))/(2 * (n_prime ** 2))
        return (e - e_prime)/const.hbar

    def __repr__(self) -> str:
        """
        String representation of the current object
        :return: string representation (str)
        """
        this_repr = "[ "
        for trans in self:
            this_repr += f"{trans}, \n"
        this_repr += "]"
        return this_repr

    def get_n_to_n_prime_couple(self) -> set:
        couple_set: set = set()
        for trans in self:
            couple_set.add(trans.get_n_to_n_prime_couple())
        return couple_set

    def intensity_of_the_beam(self, T: float, z: int = const.Z_H, mu: float = const.mu_H) -> np.ndarray:
        from QuantumFactory import QuantumFactory
        I: list = list()
        alpha = sp.Symbol('alpha')  # proportional function
        for couple in self.get_n_to_n_prime_couple():
            omega = QuantumFactory.get_transition_angular_frequency_unperturbated(couple[0], couple[1], z, mu)
            N = QuantumFactory.decay_number(couple[0], T, z, mu)
            Rs_mean = self.get_spontanious_decay_mean(z, mu)
            I.append(alpha*N*omega*Rs_mean)
        return np.array(I)

    def relative_intensity_of_the_beam(self, T: float, z: int = const.Z_H, mu: float = const.mu_H):
        I_ratio: list = list()
        for couple in self.get_n_to_n_prime_couple():
            I_ratio.append(self.intensity_of_the_beam(T, z, mu))

    def save(self):
        raise NotImplemented("This method is not implemented yet")


if __name__ == '__main__':
    n, n_prime = 6, 2

    transitions_n_to_n_prime = Transitions(n=n, n_prime=n_prime, hydrogen=True)
    print(transitions_n_to_n_prime)

