import numpy as np
import matplotlib.pyplot as plt

Vc = 2.405
lc = 620e-09
lo = 632.8e-09
NA = 0.12
a = Vc*lc/(2*np.pi*NA)
V = 2*np.pi*a*NA/lo
wLF1 = a*(0.65+1.619*V**(-1.5)+2.879*V**(-6))
c = 3e08
f = 4.5e-03
t = 1.3e-03/2
wLAS = 0.63e-03/2
alpha = (t/wLAS)**2
beta = f*lo/(2*np.pi*wLAS*wLF1)+np.pi*wLAS*wLF1/(2*f*lo)
gamma = np.pi*wLAS*wLF1*alpha/(2*f*lo)
Z = np.sqrt(((lo*f/(np.pi*wLF1))**2-wLAS**2)/((wLAS**2)*alpha))

print(alpha)
print(beta)
print(gamma)
print(Z)


def T_fct(Z):
    return (np.sqrt(1 + alpha*Z**2)/(beta+gamma*Z**2))**2


Z = np.linspace(0, 1, 10_000)
T = T_fct(Z)

plt.plot(Z, T)
plt.ylabel("T (Efficacité) [-]")
plt.xlabel("Z [m]")
plt.grid()
plt.show()
