import numpy as np
import sympy as sp
from sympy import symbols
from math import pow
from sympy.functions import exp
import matplotlib.pyplot as plt

def v0(r, a, b):
    if a < r < b:
        return np.exp(-4 * (2 * r - a - b)**2 / ((b - a) ** 2 - (2 * r - a - b)**2))
    else:
        return 0 * r

def v0rr(r, a, b, d):
    if a < r < b:
        return 1/r**(d-1)* (r**(d - 1)*(-4*(-4*a - 4*b + 8*r)/((-a + b)**2 - (-a - b + 2*r)**2) -
                4*(-4*a - 4*b + 8*r)*(-a - b + 2*r)**2/((-a + b)**2 -
                (-a - b + 2*r)**2)**2)**2*np.exp(-4*(-a - b + 2*r)**2/((-a + b)**2 -
                (-a - b + 2*r)**2)) + r**(d - 1)*(-32/((-a + b)**2 - (-a - b + 2*r)**2) -
                8*(-4*a - 4*b + 8*r)**2/((-a + b)**2 - (-a - b + 2*r)**2)**2 -
                32*(-a - b + 2*r)**2/((-a + b)**2 - (-a - b + 2*r)**2)**2 -
                4*(-8*a - 8*b + 16*r)*(-4*a - 4*b + 8*r)*(-a - b + 2*r)**2/((-a + b)**2 -
                (-a - b + 2*r)**2)**3)*np.exp(-4*(-a - b + 2*r)**2/((-a + b)**2 -
                (-a - b + 2*r)**2)) + r**(d - 1)*(d - 1)*(-4*(-4*a - 4*b + 8*r)/((-a + b)**2 -
                (-a - b + 2*r)**2) - 4*(-4*a - 4*b + 8*r)*(-a - b + 2*r)**2/((-a + b)**2 -
                 (-a - b + 2*r)**2)**2)*np.exp(-4*(-a - b + 2*r)**2/((-a + b)**2 -
                (-a - b + 2*r)**2))/r)
    else:
        return 0 * r


def solve(d, V, c, a, b, x, tsteps, h, time):
    u = np.zeros((len(time), len(x)))
    tau = time[1] - time[0]

    for i in range(len(x)):
        if ((x[i] > a) and (x[i] < b)):
            u[0, i] = v0(x[i], a, b)
            u[1, i] = u[0, i] + tau**2 / 2 * c**2 * v0rr(x[i], a, b, d)

    for n in range(1, len(time) - 1):
        u[n + 1, 1:-1] = 2 * u[n, 1:-1] - u[n - 1, 1:-1] + \
                        tau**2 * c**2 / h / x[1:-1]**(d-1) * ((x[1:-1] + h/2) ** (d-1) *
                        (u[n, 2:] - u[n, 1:-1]) / h - (x[1:-1] - h/2) ** (d-1) *
                        (u[n, 1:-1] - u[n, :-2]) / h)
        u[n+1, 0] = u[n+1, 1]
        u[n+1, -1] = u[n+1, -2]

    return u

def cnorm(u):
    return np.max(np.abs(u), axis=1)

def lnorm(u, h):
    return np.sqrt(np.sum(np.abs(u)**2, axis=1))

rmin = 0
rmax = 1.8
c = 1.5
a = 0.6
b = 1.2
f = 0
T = 3
I0 = 200
al = 3

R, A, B = symbols("r a b")
V = exp((-4 * (2 * R - (A + B))**2)/((B - A)**2 - (2 * R - (A + B))**2))

I = np.array([I0, I0*al, I0*al*al])
courant = 0.5

h = np.array([])
h = (rmax - rmin)/I

tau1 = h[0] * courant / c
tstep1 = int(T/tau1)
tsteps = tstep1 * np.array([1, 3, 9])

x = [0] * 3
time = [0] * 3
u = [0] * 3

cart = False
cyl = False
spher = True

for i in range(3):
    x[i] = np.linspace(rmin, rmax + h[i], I[i] + 2)
    x[i] -= h[i]/2
    time[i] = np.linspace(0, T, tsteps[i] + 1)
    if (cart):
        u[i] = solve(1, V, c, a, b, x[i], tsteps[i], h[i], time[i])
        title = '1d'
    elif (cyl):
        u[i] = solve(2, V, c, a, b, x[i], tsteps[i], h[i], time[i])
        title = '2d'
    elif (spher):
        u[i] = solve(3, V, c, a, b, x[i], tsteps[i], h[i], time[i])
        title = '3d'

t1 = 2
t2 = 5

c1 = cnorm(u[1][::3, t1:-1:3] - u[0][:, 1:-1])[1:]
c2 = cnorm(u[2][::9, t2:-1:9] - u[1][::3, t1:-1:3])[1:]

l1 = lnorm(u[1][::3, t1:-1:3] - u[0][:, 1:-1], h[0])[1:]
l2 = lnorm(u[2][::9, t2:-1:9] - u[1][::3, t1:-1:3], h[0])[1:]

alpha = [al*al for i in range(tsteps[0])]

plt.figure(figsize=(10,8))
plt.plot(time[0][1:], c1/c2, label='Cnorm')
plt.plot(time[0][1:], l1/l2, label='Lnorm')
plt.xlabel("Time", fontsize = 15)
plt.ylabel("Ratio", fontsize = 15)
plt.title(title, fontsize=15)
plt.plot([0, T], [9.0, 9.0], label='1/(alpha)^2')
plt.legend(fontsize=15)
plt.grid()
plt.savefig(title + ".pdf")
plt.show()

