import numpy as np
import Projet.Constantes as const

class Geometric_calculus:
    """
    This class wraps up methods that calculate multiple geometric
     information such as angles and distances
    """

    def __init__(self, w_c, angle, a, L, B, C) -> None:
        """Constructor

        Parameters
        ----------
        w_c : float
            Calibrating frequency in Hz
        angle : float
            Angle of the beam producer in rad
        a : float
            Lenght of a side of the prism in meters
        L : float
            Distance between the center of the prism and the screen in meter
        B : float
            First Cauchy law's constant
        C : float
            Second Cauchy law's constant in meters^2

        Attributes
        ----------
        self.w_c: float
            Calibrating frequency in Hz of the system
        self.angle: float
            Angle of the beam producer in rad
        self.phi : float
            Angle of the prism
        self.a: float
            Lenght of a side of the prism
        self.L: float
            Distance between the center of the prism and the screen in meter
        self.B : float
            First Cauchy law's constant
        self.C : float
            Second Cauchy law's constant in meters^2
        self._x_ref : float
            Position of the beam on the screen for the frequency w_c in meters
        """
        self._w_c = w_c
        self._angle = angle
        self._a = a
        self._L = L
        self._B = B
        self._C = C
        self._phi = self._get_phi(self._w_c)
        self._x_ref = self.get_x_for_frequency(self._w_c)

    def _get_phi(self, w_c) -> float:
        """This methods calculates the value of 
        the angle of the rotation of the prism which
        ensures that the beam with a frequency of w_c
        is perpendicular to the screan

        Parameters
        ----------
        w_c : float
            Calibrating frequency in Hz
        angle : float
            Angle of the beam producer in rad
        B : float
            First Cauchy law's constant
        C : float
            Second Cauchy law's constant in  meters^2
        Returns
        ---------
        phi: float
            value of the angle of rotation of the prism in rad
        """
        n = self._get_refraction_value(w_c)
        sin_argument = -np.arcsin((1/(2*n)))+ np.pi/3
        arcsin_argument = n*np.sin(sin_argument)
        phi = np.arcsin(arcsin_argument) - self._angle - np.pi/6
        return phi

    def _get_refraction_value(self,w) -> float:
        """This methods calculates the value of 
           the angle of the rotation of the prism which
            ensures that the beam with a frequency of w_c
            is perpendicular to the screan

            Parameters
            ----------
            w : float
                 Calibrating frequency in frequency of the beam
            Returns
            ----------
            value: float
                The value of the refraction index for the given frequency
        """
        c = const.c
        value = self._B + self._C/(((2*np.pi*c)/w)**2)
        return value

    def get_x_for_frequency(self, w) -> float:
        """This methods calculates the value of
            the position of the beam on the screen

            Parameters
            ----------
            w : float
                frequency of the beam in Hz

            Returns
            ----------
            value: float
                The value of the position of the beam on the screen in meters
        """
        x = self.get_h_of_first_refraction_of_beam(w) - self.get_differences_of_h(w)
        return x

    def get_h_of_first_refraction_of_beam(self, w) -> float:
        """This methods calculates the value of
            the position of the beam on the screen if
            it was refracted only once

            Parameters
            ----------
            w : float
                frequency of the beam in Hz

            Returns
            ----------
            value: float
                The value of the position of the beam on the screen in meters if the beam was refracted only once
        """
        n = self._get_refraction_value(w)
        arcsin_argument = np.sin(np.pi/6 + self._angle+ self._phi)/n
        h_top = (self._L + self._a/8)*(np.sin(np.arcsin(arcsin_argument) - np.pi/6))
        return h_top

    def get_differences_of_h(self, w) -> float:
        """This methods calculates the value of
            the difference of height between one and two diffraction

            Parameters
            ----------
            w : float
                frequency of the beam in Hz

            Returns
            ----------
            value: float
                the height differences in m
        """
        

    def get_delta_x(self, w, angle, a, L, B, C) -> float:
        """This methods calculates the value of
            the distance between the beam and the bean of frequency of w_c
            on the screen

            Parameters
            ----------
            w : float
                frequency of the beam in Hz
            angle : float
                Angle of the beam producer in rad
            a : float
                Lenght of a side of the prism in meters
            L : float
                Distance between the center of the prism and the screen in meter
            B : float
                First Cauchy law's constant
            C : float
                Second Cauchy law's constant in meters^2

            Returns
            ----------
            value: float
                The value of the position of the beam on the screen in meters
        """
        delta_x = self.get_x_for_frequency(w, angle, a, L, B, C) - self._x_ref
        return delta_x

if __name__ == "__main__":
    G = Geometric_calculus(1, np.pi/6, 2e-2, 50e-2, 1.4580, 0.00354e-12)