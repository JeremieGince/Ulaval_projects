from Transitions import Transitions
import Constantes as const


if __name__ == '__main__':
    rs_mean_normalized_coeff = ((const.alpha**5)*const.mu0*(const.c**2))/const.hbar
    omega_normalized_coeff = ((const.alpha ** 2) * const.mu0 * (const.c ** 2)) / (2*const.hbar)

    transitions_3_to_2 = Transitions(n=3, n_prime=2)
    print(transitions_3_to_2)
    transitions_3_to_2_rs_mean = transitions_3_to_2.get_spontanious_decay_mean() / rs_mean_normalized_coeff
    print(f"R^s_mean / rs_mean_normalized_coeff  ="
          f" {transitions_3_to_2_rs_mean}")
    print(f"omega / omega_normalized_coeff  ="
          f" {transitions_3_to_2.get_angular_frequency(3, 2) / omega_normalized_coeff}")
