from QuantumState import QuantumState
from Transition import Transition


def get_spontanious_decay_mean(n, n_prime):
    transitions = QuantumState.get_valid_transitions_n_to_n(n, n_prime)
    print(transitions)
    decay_sum = 0
    for transition in transitions:
        rs = transition.get_spontaniuous_decay_rate()
        print(f"R^s = {rs}")
        decay_sum += rs
    return decay_sum/len(transitions)


if __name__ == '__main__':
    get_spontanious_decay_mean(2, 1)
