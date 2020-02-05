import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    D: np.ndarray = np.array([0, 1.05, 2.04, 2.97, 4.09, 4.97, 6.02, 7.02, 8.02, 8.98, 10.06, 11, 12, 13.04, 14.03,
                              14.57, 15.99, 17.05, 17.98])
    A: np.ndarray = np.array([19, 19, 18.4, 18, 17.2, 16.2, 15.2, 14.6, 13.8, 12.8, 11.8, 11, 10.4, 9.5, 9.0, 8.5, 7.3,
                              6.5, 6.0])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    line, = ax.plot(D, A, '.', color='blue', lw=2, label="Données expérimentales")
    poly_start_idx = 0
    poly_end_idx = len(D)
    fitValues = np.polyfit(D[poly_start_idx: poly_end_idx], A[poly_start_idx: poly_end_idx], 1)
    fit, = ax.plot(D[poly_start_idx: poly_end_idx], np.poly1d(fitValues)(D[poly_start_idx: poly_end_idx]), color='red',
                   lw=2, label=f"Fit linéaire: {fitValues[0]:.2f}d + {fitValues[-1]:.2f}")
    ax.set_title("Atténuation de l'onde évanescente en fonction de \n la distance entre les interfaces")
    ax.set_xlabel("Distance entre les interfaces [mm]")
    ax.set_ylabel("Atténuation [dB]")

    plt.grid()
    plt.legend()
    plt.show()

