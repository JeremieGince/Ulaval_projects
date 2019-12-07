from Transitions import Transitions
import Constantes as const


if __name__ == '__main__':
    rs_mean_normalized_coeff = ((const.alpha**5)*const.mu0*(const.c**2))/const.hbar
    omega_normalized_coeff = ((const.alpha ** 2) * const.mu0 * (const.c ** 2)) / (2*const.hbar)

    transitions_2_to_1 = Transitions(n=2, n_prime=1)
    print(transitions_2_to_1)
    print(f"R^s_mean / rs_mean_normalized_coeff  ="
          f" {transitions_2_to_1.get_spontanious_decay_mean()/rs_mean_normalized_coeff}")
