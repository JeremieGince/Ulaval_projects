import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    D: np.ndarray = np.array([i for i in range(11)]+[i for i in range(12, 22, 2)])
    A: np.ndarray = np.array([14.0, 12.8, 11.9, 10.6, 9.52, 8.9, 8.2, 7.4, 7.0, 6.7, 6.2, 5.8, 5.6, 4.3, 2.8, 1.3])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    line, = ax.plot(D, A, '.', color='blue', lw=2, label="Données expérimentales")
    fitValues = np.polyfit(D[:7], A[:7], 1)
    fit, = ax.plot(D[:7], np.poly1d(fitValues)(D[:7]), color='red', lw=2,
                   label=f"Fit linéaire: {fitValues[0]:.2f}d + {fitValues[-1]:.2f}")
    ax.set_title("Atténuation de l'onde évanescente en fonction de la distance à l'interface")
    ax.set_xlabel("Distance entre l'antenne et l'interface [mm]")
    ax.set_ylabel("Atténuation [dB]")

    plt.grid()
    plt.legend()
    plt.show()

