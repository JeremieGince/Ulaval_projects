from Transitions import Transitions
from QuantumFactory import QuantumFactory
import Constants as const
from geometric_calculus import Geometric_calculus
import numpy as np
import time


def tab_cell(n, n_prime):
    rs_answ = {"(3, 2)": 0.00274668657777777777777777777778,
               "(4, 2)": 0.00052436,
               "(5, 2)": 0.000157596,
               "(6, 2)": 0.000060611}

    om_answ = {"(3, 2)": 5/36,
               "(4, 2)": 3/16,
               "(5, 2)": 21/100,
               "(6, 2)": 2/9}

    rs_mean_normalized_coeff = ((const.alpha ** 5) * const.mu_H * (const.c ** 2)) / const.hbar
    omega_normalized_coeff = ((const.alpha ** 2) * const.mu_H * (const.c ** 2)) / (2 * const.hbar)

    print(f'-'*75)

    print(f"Transition ({n} -> {n_prime}) : \n")

    transitions_n_to_n_prime = Transitions(n=n, n_prime=n_prime, hydrogen=True)
    # print(f"Transitions: {transitions_n_to_n_prime} \n")

    transitions_n_to_n_prime_rs_mean = transitions_n_to_n_prime.get_spontaneous_decay_mean() / rs_mean_normalized_coeff
    print(f"R^s_mean / rs_mean_normalized_coeff  = {transitions_n_to_n_prime_rs_mean:.5e}")

    reel_rs_mean = rs_answ[str((n, n_prime))] if str((n, n_prime)) in rs_answ else 0.0

    print(f"reference R^s  / rs_mean_normalized_coeff = {reel_rs_mean:.5e} \n")

    omega = QuantumFactory.get_transition_angular_frequency_unperturbed(n, n_prime, const.Z_H, const.mu_H)
    omega_normalized = omega / omega_normalized_coeff

    print(f"omega / omega_normalized_coeff  =  {omega_normalized:.5e}")

    reel_omega_mean = om_answ[str((n, n_prime))] if str((n, n_prime)) in om_answ else 0.0

    print(f"reference omega  / omega_normalized_coeff = {reel_omega_mean:.5e}")

    print(f'-' * 75)


if __name__ == '__main__':
    problem_variable = {
        "theta": np.pi/4,  # [rad]
        "a": 2e-2,  # [m]
        "L": 50e-2,  # [m]
        "B": 1.4580,  # [?]
        "C": 0.00354e-12,  # [m^2]
        "T": 1_000,  # [K]
    }
    couples = [(3, 2), (4, 2), (5, 2), (6, 2)]

    start_time = time.time()

    # Problem 2.a
    print("\n Problem 2.a \n")
    for transition in couples:
        tab_cell(transition[0], transition[1])

    # Problem 2.b
    print("\n Problem 2.b \n")
    omega_c = QuantumFactory.get_transition_angular_frequency_unperturbed(4, 2, const.Z_H, const.mu_H)
    geometric_engine = Geometric_calculus(omega_c,
                                          problem_variable["theta"], problem_variable["a"], problem_variable["L"],
                                          problem_variable["B"], problem_variable["C"])

    for transition in couples:
        relative_intensity = QuantumFactory.relative_intensity_of_the_beam(transition[0], transition[1],
                                                                           4, 2, T=problem_variable["T"])
        print(f"---")
        print(f" I_{transition} / I_(4, 2) = {relative_intensity:.2e} [-]")
        omega = QuantumFactory.get_transition_angular_frequency_unperturbed(transition[0], transition[1], const.Z_H, const.mu_H)
        print(f"x position {transition}: {geometric_engine.get_delta_x(omega):.2e} [m]")
        print(f"---")

    # Problem 2.c
    print("\n Problem 2.c \n")
    # Balmer series (n' = 2)
    # refrence: https://fr.wikipedia.org/wiki/Spectre_de_l%27atome_d%27hydrog%C3%A8ne?fbclid=IwAR1Yy7gA_6GgFECMvbQgK51SCjNKGy3UQS7OM1EWvxUKtHODyz-0LM_Azjk
    Balmer_series = [(3, 2), (4, 2), (5, 2), (6, 2)]

    for transition in Balmer_series:
        relative_intensity = QuantumFactory.relative_intensity_of_the_beam(transition[0], transition[1],
                                                                           4, 2, T=problem_variable["T"])
        print(f"---")
        print(f" I_{transition} / I_(4, 2) = {relative_intensity:.2e} [-]")
        omega = QuantumFactory.get_transition_angular_frequency_unperturbed(transition[0], transition[1], const.Z_H, const.mu_H)
        print(f"x position {transition}: {geometric_engine.get_delta_x(omega):.2e} [m]")
        print(f"---")

    print(f"--- elapse time : {time.time() - start_time:.2f} s ---")

