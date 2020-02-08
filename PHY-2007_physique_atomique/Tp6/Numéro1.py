import numpy as np

def obtenir_énergie(n,l,j):
    math = 2*n + l - 1/2 - 0.0225*l*(l+1)- (0.05*(-1)**(j-l-1/2))*(2*l - j + 1/2)
    return math
"""
li = []
for n in range(1,5):
    print(n)
    for l in range(0,(8-2*n)+1):
        if l == 0:
            j = 1
            li.append((obtenir_énergie(n, l, j - 1 / 2), n, l, j - 1 / 2))
        else:
            for j in range(l,l+2):
                li.append((obtenir_énergie(n,l,j-1/2),n,l,j-1/2))
li.sort()
print("begin{align*}")
for i in li:
    print(f"{i}")
print("begin{align*}")

"""

def calcul_E_liaison(z,a):
    a_v= 15.75
    a_s= 17.80
    a_c= 0.711
    a_a =94.780
    a_p= 11.18
    rep =a_v*a - a_s*(a**(2/3)) - a_c*(z**2)/(a**(1/3)) - (a_a/a)*(z-a/2)**2 + a_p*((-1)**(z)+(-1)**(a-z))/(2*(a)**(1/2))
    return rep

def calculer_taux_alpha(A, Z, E):
    c = 2.998e8
    Z_a = 2
    r = 1.2e-15
    a = 1/137.035999
    m_a =3.72738e9
    coef = np.sqrt(E/(2*m_a))*(c/(r*(A**(1/3))))
    exp = -np.pi*(Z-Z_a)*a*Z_a*np.sqrt((2*m_a)/E)
    print(exp)
    return coef*(np.e**(exp))

print(calcul_E_liaison(6,13)-calcul_E_liaison(6,12) )