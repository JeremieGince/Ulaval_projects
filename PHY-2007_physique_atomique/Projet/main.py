from Transitions import Transitions
import Constantes as const
import numpy as np


if __name__ == '__main__':
    rs_mean_normalized_coeff = ((const.alpha**5)*const.mu_H*(const.c**2))/const.hbar
    omega_normalized_coeff = ((const.alpha ** 2) * const.mu_H * (const.c ** 2)) / (2*const.hbar)

    n, n_prime = 3, 2

    transitions_n_to_n_prime = Transitions(n=n, n_prime=n_prime)
    print(transitions_n_to_n_prime)

    transitions_n_to_n_prime_rs_mean = transitions_n_to_n_prime.get_spontanious_decay_mean() / rs_mean_normalized_coeff
    print(f"R^s_mean / rs_mean_normalized_coeff  ="
          f" {transitions_n_to_n_prime_rs_mean}")
    reel_rs_3_to_2_mean = 0.00274668657777777777777777777778
    print(f"reel R^s = {reel_rs_3_to_2_mean},"
          f" deviation: {100 * (np.abs(transitions_n_to_n_prime_rs_mean - reel_rs_3_to_2_mean) / rs_mean_normalized_coeff)} %")

    print(f"omega / omega_normalized_coeff  ="
          f" {transitions_n_to_n_prime.get_angular_frequency(3, 2, const.Z_H, const.mu_H) / omega_normalized_coeff}")
