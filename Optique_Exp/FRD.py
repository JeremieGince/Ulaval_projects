import numpy as np
import matplotlib.pyplot as plt


def fitlin(X, Y, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    line, = ax.plot(X, Y, 'o', color='blue', lw=3, label=kwargs.get("line_label", rf"Données expérimentales"))
    fitValues = np.polyfit(X, Y, 1)
    fit, = ax.plot(X, np.poly1d(fitValues)(X), color='red', lw=2,
                   label=f"Fit linéaire: y = {fitValues[0]:.3f}x + {fitValues[-1]:.3f}")

    ax.set_title(kwargs.get("title", "Graph"))
    ax.set_xlabel(kwargs.get("xlabel", "X"))
    ax.set_ylabel(kwargs.get("ylabel", "Y"))

    plt.grid()
    plt.legend()
    plt.savefig(f"{kwargs.get('title', 'Graph').replace(' ', '_').replace('é', 'e').replace('$', '')}.png", dpi=300)
    # plt.show()
    plt.close(fig)
    return fitValues


def etape1(verbose=True):
    print(f"--- Étape 1 ---") if verbose else None
    I_bobine = np.array([
        -4, -3, -2, -1, 0, 1, 2, 3, 4
    ])
    B = np.array([
        -263, -197, -130, -66, 0, 65, 132, 200, 259
    ])
    fitValues = fitlin(I_bobine, B, title="Densité de flux magnétique B [gauss] en fonction de $I_{bob} [A]$",
           xlabel="$I_{bob} [A]$", ylabel="B [gauss]")

    N = 12*235
    ell = 54
    print(f"Pente expérimentale: {fitValues[0]:.3f} [gauss/A]") if verbose else None
    print(f"Pente théorique: {0.4*np.pi*N/ell:.3f} [gauss/A]") if verbose else None

    print(f"-"*120) if verbose else None
    return fitValues[0]


def etape2(verbose=True):
    print(f"--- Étape 2 ---") if verbose else None

    liquide_A_data = {
        "lambda [nm]": 632.8,
        "matériau": "liquide A",
        "longueur de cellule [cm]": 54,
        "theta min [deg]": 90,
        "theta max [deg]": 180,
        "I max [nA]": 30_000,
        "I_bob [A]": np.array([
            -4, -3, -2, 0, 2, 3, 4
        ]),
        "I_photodiode [nA]": np.array([
            483.90, 272.84, 121.47, 0.00, 121.47, 272.84, 483.90
        ]),
        "title": "Malus, laser rouge, liquide A",
        "color": "green",
    }

    liquide_B_data = {
        "lambda [nm]": 632.8,
        "matériau": "liquide B",
        "longueur de cellule [cm]": 54,
        "theta min [deg]": 90,
        "theta max [deg]": 180,
        "I max [nA]": 30_000,
        "I_bob [A]": np.array([
            -4, -3, -2, 0, 2, 3, 4
        ]),
        "I_photodiode [nA]": np.array([
            299.35, 168.63, 75.03, 0.00, 75.03, 168.63, 299.35
        ]),
        "title": "Malus, laser rouge, liquide B",
        "color": "blue",
    }

    liquide_C_data = {
        "lambda [nm]": 632.8,
        "matériau": "liquide C",
        "longueur de cellule [cm]": 20,
        "theta min [deg]": 90,
        "theta max [deg]": 180,
        "I max [nA]": 30_000,
        "I_bob [A]": np.array([
            -4, -3, -2, 0, 2, 3, 4
        ]),
        "I_photodiode [nA]": np.array([
            426.17, 240.22, 106.92, 0.00, 106.92, 240.22, 426.17
        ]),
        "title": "Malus, laser rouge, liquide C",
        "color": "purple",
    }

    data = [liquide_A_data, liquide_B_data, liquide_C_data]

    BIpente = etape1(False)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for liquide in data:
        Y = liquide["I_photodiode [nA]"] / (liquide["I max [nA]"])
        X = liquide["I_bob [A]"] * BIpente
        line, = ax.plot(X, Y, 'o', lw=3, color=liquide["color"],  label=f'Données expérimentales {liquide["matériau"]}')
        fitValues = np.polyfit(X, Y, 2)
        fit, = ax.plot(X, np.poly1d(fitValues)(X), lw=2, color=liquide["color"],
                       label=f"Fit: $y = {fitValues[0]:.3e}x^2 + {fitValues[1]:.3e}x + {fitValues[-1]:.3e}$ pour {liquide['matériau']}")

    ax.set_title("$I_{photodiode}/I_{max}$ en fonction de B [gauss] pour les liquides A, b et C")
    ax.set_ylabel("$I_{photodiode}/I_{max}$")
    ax.set_xlabel("B [gauss]")

    plt.grid()
    plt.legend()
    # plt.savefig(f"{'I_photodiode normalisé en fonction de B [gauss] pour les liquides A, b et C'.replace(' ', '_').replace('é', 'e').replace('$', '')}.png", dpi=300)
    plt.show()
    plt.close(fig)
    print(f"-" * 120) if verbose else None


def etape3(verbose=True):
    print(f"--- Étape 3 ---") if verbose else None

    liquide_A_data = {
        "lambda [nm]": 632.8,
        "matériau": "liquide A",
        "longueur de cellule [cm]": 54,
        "I_bob [A]": np.array([
            -4, -3, -2, 0, 2, 3, 4
        ]),
        "theta analyseur [deg]": np.array([
            82.70, 84.53, 86.35, 90.00, 93.65, 95.47, 97.30
        ]),
        "title": "Verdet, laser rouge, liquide A",
        "color": "green",
    }

    liquide_B_data = {
        "lambda [nm]": 632.8,
        "matériau": "liquide B",
        "longueur de cellule [cm]": 54,
        "I_bob [A]": np.array([
            -4, -3, -2, 0, 2, 3, 4
        ]),
        "theta analyseur [deg]": np.array([
            84.27, 85.70, 87.13, 90.00, 92.87, 94.30, 95.73
        ]),
        "title": "Verdet, laser rouge, liquide B",
        "color": "blue",
    }

    liquide_C_data = {
        "lambda [nm]": 632.8,
        "matériau": "liquide C",
        "longueur de cellule [cm]": 20,
        "I_bob [A]": np.array([
            -4, -3, -2, 0, 2, 3, 4
        ]),
        "theta analyseur [deg]": np.array([
            83.15, 84.87, 86.58, 90.00, 93.42, 95.13, 96.85
        ]),
        "title": "Verdet, laser rouge, liquide C",
        "color": "purple",
    }

    data = [liquide_A_data, liquide_B_data, liquide_C_data]
    BIpente = etape1(False)

    re_data = dict()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for liquide in data:
        Y = liquide["theta analyseur [deg]"]
        X = liquide["I_bob [A]"] * BIpente
        line, = ax.plot(X, Y, 'o', lw=3, color=liquide["color"],  label=f'Données expérimentales {liquide["matériau"]}')
        fitValues = np.polyfit(X, Y, 1)
        fit, = ax.plot(X, np.poly1d(fitValues)(X), lw=2, color=liquide["color"],
                       label=rf"Fit linéaire: y = {fitValues[0]:.3f}x + {fitValues[-1]:.3f} pour {liquide['matériau']}")

        re_data[f"Pente pour {liquide['matériau']}"] = fitValues[0]
        re_data[f"Verdet pour {liquide['matériau']}"] = fitValues[0]/liquide["longueur de cellule [cm]"]

    ax.set_title(r"$\theta$ en fonction de B pour les liquides A, b et C")
    ax.set_xlabel("B [gauss]")
    ax.set_ylabel(r"$\theta \; [{}^o]$ ")

    plt.grid()
    plt.legend()
    # plt.savefig(f"{'etape3graphe'.replace(' ', '_').replace('é', 'e').replace('$', '')}.png", dpi=300)
    plt.show()
    plt.close(fig)

    for d, v in re_data.items():
        print(f"{d} = {v:.3e}")

    print(f"-" * 120) if verbose else None
    return re_data


def etape4(verbose=True):
    print(f"--- Étape 4 ---") if verbose else None

    liquide_A_data = {
        "lambda [nm]": 532,
        "matériau": "liquide A",
        "longueur de cellule [cm]": 54,
        "I_bob [A]": np.array([
            -4, -3, -2, 0, 2, 3, 4
        ]),
        "theta analyseur [deg]": np.array([
            80.42, 82.82, 85.21, 90.00, 94.79, 97.18, 99.58
        ]),
        "title": "Verdet, laser rouge, liquide A",
        "color": "green",
    }

    liquide_B_data = {
        "lambda [nm]": 532,
        "matériau": "liquide B",
        "longueur de cellule [cm]": 54,
        "I_bob [A]": np.array([
            -4, -3, -2, 0, 2, 3, 4
        ]),
        "theta analyseur [deg]": np.array([
            81.83, 83.88, 85.92, 90.00, 94.08, 96.12, 98.17
        ]),
        "title": "Verdet, laser rouge, liquide B",
        "color": "blue",
    }

    liquide_C_data = {
        "lambda [nm]": 532,
        "matériau": "liquide C",
        "longueur de cellule [cm]": 20,
        "I_bob [A]": np.array([
            -4, -3, -2, 0, 2, 3, 4
        ]),
        "theta analyseur [deg]": np.array([
            80.97, 83.23, 85.48, 90.00, 94.52, 96.77, 99.03
        ]),
        "title": "Verdet, laser rouge, liquide C",
        "color": "purple",
    }

    data = [liquide_A_data, liquide_B_data, liquide_C_data]
    BIpente = etape1(False)

    re_data = dict()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for liquide in data:
        Y = liquide["theta analyseur [deg]"]
        X = liquide["I_bob [A]"] * BIpente
        line, = ax.plot(X, Y, 'o', lw=3, color=liquide["color"],  label=f'Données expérimentales {liquide["matériau"]}')
        fitValues = np.polyfit(X, Y, 1)
        fit, = ax.plot(X, np.poly1d(fitValues)(X), lw=2, color=liquide["color"],
                       label=rf"Fit linéaire: y = {fitValues[0]:.3f}x + {fitValues[-1]:.3f} pour {liquide['matériau']}")

        re_data[f"Pente pour {liquide['matériau']}"] = fitValues[0]
        re_data[f"Verdet pour {liquide['matériau']}"] = fitValues[0]/liquide["longueur de cellule [cm]"]

    ax.set_title(r"$\theta$ en fonction de B pour les liquides A, b et C")
    ax.set_xlabel("B [gauss]")
    ax.set_ylabel(r"$\theta \; [{}^o]$ ")

    plt.grid()
    plt.legend()
    # plt.savefig(f"{'etape4graphe'.replace(' ', '_').replace('é', 'e').replace('$', '')}.png", dpi=300)
    plt.show()
    plt.close(fig)

    for d, v in re_data.items():
        print(f"{d} = {v:.3e}")

    print(f"-" * 120) if verbose else None
    return re_data


if __name__ == '__main__':
    etape1()
    etape2()
    etape3()
    etape4()
