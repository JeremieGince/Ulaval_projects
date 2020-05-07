from pathlib import Path  # Permet de gérer les emplacements exacts des fichiers

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import *

""" Ce script fait la transformée de Fourier d'un fichier de données obtenu lors du laboratoire de l'Interféromètre
de Michelson fait dans le cours PHY-2006 - Optique expérimentale.
Le code est adapté de celui de Daniel Côté dispobile à l'adresse:
https://github.com/dccote/Enseignement/tree/master/ATELIER/FFT
J'ai changé la fonction d'extraction de données qui était utilisée auparavant pour la fonction genfromtxt de numpy qui
fonctionne mieux selon moi. J'ai ajouté un facteur 2 au dx de la fonction fourierTransformInterferogram pour prendre
en compte le trajet deux fois plus long de la lumière par rapport au déplacement du miroir M1.
J'ai aussi ajouté une petite fonction plot_ocean_optics qui permet de lire et d'afficher les données du spectrogramme
Ocean Optics utilisé en laboratoire comme référence.
"""
Directory: str = "Données_Michelson_2/Données_Michelson/"


def read_from_file(filename):
    file = np.genfromtxt(str(Path.cwd() / Directory / filename), skip_header=18, skip_footer=1)
    x = file[:, 1]
    y = file[:, 2]
    return (x, y)  # J: J'ai chamgé la fonction np.loadtxt par la fonction np.genfromtxt (plus efficace)


def plot_ocean_optics(filename, title, left, right):  # Il les fichiers sont préalablement modifiés pour changer
    # toutes les virgules en points.(ctrl+f) Normalement il est préférable d'enlever les header et footer avec la fonction
    file = np.genfromtxt(str(Path.cwd() / Directory / filename))
    plt.figure(figsize=(10, 7))
    plt.title("{}".format(title))
    plt.xlabel("Longueur d'onde [nm]")
    plt.ylabel("Intensité [-]")
    plt.xlim(left=left, right=right)
    plt.plot(file[:, 0], file[:, 1])
    plt.savefig(f"FiguresIM/{title.replace(' ', '').replace(',', '').replace('.', '').replace('è', 'e')}.png", dpi=300)




plot_ocean_optics("OceanOptics_spectres_de_référence/"+"autre_lampe_sodium_faible.txt", "Ocean Optics, lampe sodium faible", 400, 700)
# plot_ocean_optics("OceanOptics_spectres_de_référence/"+"autre_lampe_sodium_forte.txt", "Ocean Optics lampe sodium forte", 400, 900)
plot_ocean_optics("OceanOptics_spectres_de_référence/"+"autre_Lazer_He_Ne.txt", "Ocean Optics, laser He-Ne", 400, 700)
plot_ocean_optics("OceanOptics_spectres_de_référence/"+"autre_lumiere_blanche.txt", "Ocean Optics, lumiere blanche", 200, 900)
# plot_ocean_optics("OceanOptics_spectres_de_référence/"+"oceanHG.txt", "Ocean Optics merccure", 200, 700)
# plot_ocean_optics("OceanOptics_spectres_de_référence/"+"oceanjaune.txt", "Ocean Optics blanche filtre jaune", 400, 700)


def fourierTransformInterferogram(x, y):
    """ A partir du tableau de valeurs Y correspondant a l'abscisse X,
    la transformée de Fourier est calculée et l'axes des fréquences (f en
    µm^-1) et des wavelengths (1/f en microns) est retournée.
    Le spectre est un ensemble de valeurs complexes pour lesquelles l'amplitude
    et la phase sont pertinentes: l'ordre des valeurs commence par la valeur DC (0)
    et monte jusqu'a f_max=1/2/∆x par resolution de ∆f = 1/N/∆x. A partir de la
    (N/2) ieme valeur, la frequence est negative jusqu'a -∆f dans la N-1 case.
    Voir
    https://github.com/dccote/Enseignement/blob/master/HOWTO/HOWTO-Transformes%20de%20Fourier%20discretes.pdf
    """
    spectrum = fft(y)
    dx = 2 * (x[1] - x[
        0])  # on obtient dx, on suppose equidistant J: ajout d'un facteur 2 pour le déplacement du miroir
    N = len(x)  # on obtient N directement des données
    frequencies = fftfreq(N, dx)  # Cette fonction est fournie par numpy
    wavelengths = 1 / frequencies - float(200e-9)  # Les fréquences en µm^-1 sont moins utiles que lambda en µm
    return (wavelengths, frequencies, spectrum)


def plotCombinedFigures(x, y, w, s, title="", left=400, right=800):
    """"
    On met l'interferogramme et le spectre sur la meme page.
    """
    fig, (axes, axesFFT) = plt.subplots(2, 1, figsize=(10, 10))
    plt.subplots_adjust(hspace=0.31)
    axes.plot(x, y, '-')
    axes.set_title("Interferogramme")
    axes.set_xlabel("Déplacement du miroir 1 [µm]")
    axes.set_ylabel("Voltage [mV]")
    axesFFT.plot(w * 1000, abs(s), '.-')
    axesFFT.set_xlim(left=left, right=right)
    axesFFT.set_xlabel("Longueur d'onde [nm]")
    axesFFT.set_ylabel("Intensité [-]")
    axesFFT.set_title(title)
    plt.savefig(f"FiguresIM/{title.replace(' ', '').replace(',', '').replace('.', '').replace('è', 'e')}.png", dpi=300)


def graph_mesure_et_fft(nom_fichier, lim_gauche, lim_droite, titre):
    (x, y) = read_from_file(nom_fichier)
    (w, f, s) = fourierTransformInterferogram(x, y)
    df = f[1] - f[0]
    dl = 0.500 * 0.500 * df * 1000
    print(f"{titre}, dl: {dl}, N:{len(x)}")
    plotCombinedFigures(x, y, w, s, left=lim_gauche, right=lim_droite,
                        title="{1}, resolution {0:0.2f} [nm]".format(dl, titre))


# graph_mesure_et_fft("laser_he_ne_3.txt", 300, 800, "Laser He-Ne")
# graph_mesure_et_fft("lum_blanche_6.txt", 0, 1300, "Lumière blanche")
graph_mesure_et_fft("lum_sodium_1.txt", None, None, "Lumière sodium longue mesure")
# graph_mesure_et_fft("lum_sodium_2.txt", 200, 1000, "Lumière sodium courte mesure")

plt.show()  # J: À la fin pour afficher tous les graphiques en même temps
