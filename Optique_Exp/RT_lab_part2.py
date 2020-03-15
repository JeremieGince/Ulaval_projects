import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    D_0: np.ndarray = np.array([0, 1.05, 2.04, 2.97, 4.09, 4.97, 6.02, 7.02, 8.02, 8.98, 10.06, 11, 12, 13.04, 14.03,
                                14.57, 15.99, 17.05, 17.98])
    A_0: np.ndarray = np.array([19, 19, 18.4, 18, 17.2, 16.2, 15.2, 14.6, 13.8, 12.8, 11.8, 11, 10.4, 9.5, 9.0, 8.5, 7.3,
                                6.5, 6.0])

    D_1: np.ndarray = np.array([0, 1.09, 2.01, 2.98, 3.98, 5.02, 6.06, 6.99, 7.99, 9.01, 10.03, 11.02, 12.04, 13.02,
                               14.02, 14.99, 16.07, 17.02, 18.06])
    A_1: np.ndarray = np.array([19, 19, 18.4, 17.8, 17, 16, 15.4, 14.6, 13.8, 12.8, 12, 11, 10.2, 9.5, 8.7, 7.9,
                               7.2, 6.5, 5.7])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    line_0, = ax.plot(D_0, A_0, '.', color='blue', lw=2, label="Données expérimentales - exp 0")
    line_1, = ax.plot(D_1, A_1, '.', color='green', lw=2, label="Données expérimentales - exp 1")

    poly_start_idx, poly_end_idx = 3, len(D_0)

    fitValues_0 = np.polyfit(D_0[poly_start_idx: poly_end_idx], A_0[poly_start_idx: poly_end_idx], 1)
    Attenuation_values_0 = np.poly1d(fitValues_0)(D_0[poly_start_idx: poly_end_idx])

    fitValues_1 = np.polyfit(D_1[poly_start_idx: poly_end_idx], A_1[poly_start_idx: poly_end_idx], 1)
    Attenuation_values_1 = np.poly1d(fitValues_1)(D_1[poly_start_idx: poly_end_idx])

    fit_0, = ax.plot(D_0[poly_start_idx: poly_end_idx], Attenuation_values_0, "-.",
                     lw=2, label=f"Fit linéaire - Exp 0: {fitValues_0[0]:.2f}d + {fitValues_0[-1]:.2f}")

    fit_1, = ax.plot(D_1[poly_start_idx: poly_end_idx], Attenuation_values_1, "-",
                     lw=2, label=f"Fit linéaire - Exp 1: {fitValues_0[0]:.2f}d + {fitValues_1[-1]:.2f}")

    D_error, A_error = np.abs(D_1 - D_0), np.abs(A_1 - A_0)
    plt.errorbar(D_0, A_0, xerr=D_error, yerr=A_error, linestyle=' ')

    ax.set_title("Atténuation de l'onde évanescente en fonction de \n la distance entre les interfaces")
    ax.set_xlabel("Distance entre les interfaces [mm]")
    ax.set_ylabel("Atténuation [dB]")

    plt.grid()
    plt.legend()
    plt.show()
    plt.close(fig)

