import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    Alpha: np.ndarray = np.array([0, 15, 30, 45, 60, 75, 90])
    Gamma: np.ndarray = np.array([0, 24, 31, 53, -73, -79, -95])
    ABratio: np.ndarray = np.array([np.infty, 6.25, 2.97, 2.29, 2.6, 4.94, 196.5])

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)

    line, = ax.plot(Alpha, -np.abs(Gamma), '.', color='blue', lw=2, label=rf"Données expérimentales")
    ax.set_title(fr"1 prisme, $n_1 = 1.53$")
    # ax.set_xlabel(r" $\alpha [{}^\degree]$")
    ax.set_ylabel(r" $\gamma \; [{}^\degree]$")
    plt.grid()
    plt.legend()

    ax_1 = fig.add_subplot(2, 1, 2)
    line_1, = ax_1.plot(Alpha, np.abs(20*np.log10(ABratio)), '.', color='blue', lw=2, label=rf"Données expérimentales")
    # ax_1.set_title(fr"1 prisme, $n_1 = 1.53$")
    ax_1.set_xlabel(r" $\alpha \; [{}^\degree]$")
    ax_1.set_ylabel(r" $abs(20\log({\frac{a}{b}})) \; [dB]$")

    plt.grid()
    plt.legend()
    plt.show()
    plt.close(fig)

    # 7.2
    def getDelta(gamma, alpha):
        return np.arccos(np.tan(2*gamma)/np.tan(2*alpha))

    delta = getDelta(np.deg2rad(np.abs(Gamma[1:-1])), np.deg2rad(Alpha[1:-1])).mean()
    print(f"delta mean exp: {delta}")  # On voit que le résultat diverge

