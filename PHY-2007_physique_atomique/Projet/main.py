from Transitions import Transitions
from QuantumFactory import QuantumFactory
import Constantes as const
import numpy as np


def tab_cell(n, n_prime):
    rs_answ = {"(3, 2)": 0.000060611,
               "(4, 2)": 0.00052436,
               "(5, 2)": 0.000157596,
               "(6, 2)": 0.00274668657777777777777777777778}

    om_answ = {"(3, 2)": 5/36,
               "(4, 2)": 3/16,
               "(5, 2)": 21/100,
               "(6, 2)": 2/9}

    rs_mean_normalized_coeff = ((const.alpha ** 5) * const.mu_H * (const.c ** 2)) / const.hbar
    omega_normalized_coeff = ((const.alpha ** 2) * const.mu_H * (const.c ** 2)) / (2 * const.hbar)

    print(f'-'*175)

    transitions_n_to_n_prime = Transitions(n=n, n_prime=n_prime, hydrogen=True)
    # print(f"Transitions: {transitions_n_to_n_prime} \n")

    transitions_n_to_n_prime_rs_mean = transitions_n_to_n_prime.get_spontanious_decay_mean() / rs_mean_normalized_coeff
    print(f"R^s_mean / rs_mean_normalized_coeff  = {transitions_n_to_n_prime_rs_mean:.5e}")

    reel_rs_mean = rs_answ[str((n, n_prime))] if str((n, n_prime)) in rs_answ else 0.0

    print(f"reel R^s = {reel_rs_mean:.5e} \n")

    omega = QuantumFactory.get_transition_angular_frequency_unperturbated(n, n_prime, const.Z_H, const.mu_H)
    omega_normalized = omega / omega_normalized_coeff

    print(f"omega / omega_normalized_coeff  =  {omega_normalized:.5e}")

    reel_omega_mean = om_answ[str((n, n_prime))] if str((n, n_prime)) in om_answ else 0.0

    print(f"reel omega = {reel_omega_mean:.5e}")

    print(f'-' * 175)


if __name__ == '__main__':
    problem_variable = {
        "theta": np.pi/4,  # [rad]
        "a": 2e-2,  # [m]
        "L": 50e-2,  # [m]
        "B": 1.4580,
        "C": 0.00354e-6,  # [m^2]
        "T": 1_000,  # [K]
    }
    couples = [(3, 2), (4, 2), (5, 2), (6, 2)]
    # Problem 2.a
    for couple in couples:
        tab_cell(couple[0], couple[1])

    # Problem 2.b
    for couple in couples:
        QuantumFactory.ratio_intensity_of_the_beam(couple[0], couple[1], 4, 2, T=problem_variable["T"])
