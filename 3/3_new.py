import numpy as np
import sympy as sp
from sympy import symbols, re
from math import pow
from sympy.functions import exp
import matplotlib.pyplot as plt

def U(x, a, b):
    x *= -1
    if a < x < b:
        return np.exp(-4 * (2 * x - a - b)**2 / ((b - a) ** 2 - (2 * x - a - b)**2))
    else:
        return 0 * x

uuuu = np.vectorize(U, excluded={'a', 'b'})

def _dudx(x, a, b):
    x *= -1
    if a < x < b:
        return (-4*(-4*a - 4*b + 8*x)/((-a + b)**2 - (-a - b + 2*x)**2) - 4*(-4*a - 4*b + 8*x)*(-a - b + 2*x)**2/((-a + b)**2 - (-a - b + 2*x)**2)**2)*np.exp(-4*(-a - b + 2*x)**2/((-a + b)**2 - (-a - b + 2*x)**2))
    else:
        return 0*x
dudx = np.vectorize(_dudx, excluded={'a', 'b'})

def U0(r, d, a, b):
    return r**((1 - d) / 2) * U(-r, a, b)

def U1(r, d, ctau, a, b):
    return r**((1 - d) / 2) * U(ctau - r, a, b)

def dUr(x, r, d, a, b):
    return -(1 - d) / 2 * r ** (-(1+d)/2) * U(x, a, b) - r ** ((1-d)/2) * dudx(x, a, b)

def f(t, r, d, c, a, b):
    if d != 2:
        return 0*r
    else:
        return c**2 * (d-1)*(d-3)/4/r**((d+3)/2)*uuuu(c*t - r, a, b)


def solve(d, c, a, b, x, tsteps, h, time):
    u = np.zeros((len(time), len(x)))

    tau = time[1] - time[0]

    r_min = (x[1] + x[0]) / 2
    r_max = (x[-1] + x[-2]) / 2

    h = x[3] - x[2]
    for i in range(len(x)):
        u[0, i] = U0(x[i], d, a, b)
        u[1, i] = U1(x[i], d, c*tau, a, b)

    for n in range(1, time.size - 1):
        u[n + 1, 1:-1] = 2 * u[n, 1:-1] - u[n - 1, 1:-1] + tau**2 * c**2 / h / x[1:-1]**(d-1) * ((x[1:-1] + h/2) ** (d-1) *(u[n, 2:] - u[n, 1:-1]) / h - (x[1:-1] - h/2) ** (d-1) * (u[n, 1:-1] - u[n, :-2]) / h) + tau**2 * f(time[n], x[1:-1], d, c, a, b)
        u[n+1, 0] = u[n+1, 1] + h * dUr(c*time[n+1] - r_min, r_min, d, a, b)
        u[n+1, -1] = u[n+1, -2] - h * dUr(c*time[n+1] - r_max, r_max, d, a, b)

    return u

def cnorm(u):
    return np.max(np.abs(u), axis=1)

def lnorm(u, h):
    return np.sqrt(np.sum(np.abs(u)**2, axis=1))

rmin = 0.1
rmax = 1.9
c = 1.6
a = 0.7
b = 1.3
T = 0.3
I0 = 200
al = 3

I = np.array([I0, I0*al])
courant = 0.01

h = np.array([])
h = (rmax - rmin)/I

tau1 = h[0] * courant / c
tstep1 = int(T/tau1)
tsteps = tstep1 * np.array([1, 3])

x = [0] * 2
time = [0] * 2
u = [0] * 2

cart = False
cyl = True
spher = False

rsol = [1] * 2

for i in range(2):
    x[i] = np.linspace(rmin, rmax + h[i], I[i] + 2)
    x[i] -= h[i]/2
    time[i] = np.linspace(0, T, tsteps[i] + 1)
    if (cart):
        u[i] = solve(1, c, a, b, x[i], tsteps[i], h[i], time[i])
        title = '1d'
        d = 1
    elif (cyl):
        u[i] = solve(2, c, a, b, x[i], tsteps[i], h[i], time[i])
        title = '2d'
        d = 2
    elif (spher):
        u[i] = solve(3, c, a, b, x[i], tsteps[i], h[i], time[i])
        title = '3d'
        d = 3
    gr, gt = np.meshgrid(x[i], time[i])
    p = c*gt - gr
    rsol[i] = gr ** ((1 - d) / 2) * uuuu(p, a, b)

U_teor = [1] * 2
for i in range(2):
    U_ = np.zeros((len(time[i]), len(x[i])))
    for t in range(tsteps[0]):
        for r in range(len(x[0])):
            U_[t, r] = x[0][r]**(int((1-d)/2)) * U(c*time[i][t] - x[i][r], a, b)
    U_teor[i] = U_

"""c1 = cnorm(u[0][:, 1:-1] - U_teor[0][:, 1:-1])[1:]
c2 = cnorm(u[1][:, 1:-1] - U_teor[1][:, 1:-1])[3::3]

l1 = lnorm(u[0][:, 1:-1] - U_teor[0][:, 1:-1], h[0])[1:]
l2 = lnorm(u[1][:, 1:-1] - U_teor[1][:, 1:-1], h[0])[3::3]"""

c1 = cnorm(u[0][:, 1:-1] - rsol[0][:, 1:-1])[1:]
c2 = cnorm(u[1][:, 1:-1] - rsol[1][:, 1:-1])[3::3]

l1 = lnorm(u[0][:, 1:-1] - rsol[0][:, 1:-1], h[0])[1:]
l2 = lnorm(u[1][:, 1:-1] - rsol[1][:, 1:-1], h[0])[3::3]

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
#plt.savefig(title + ".pdf")
plt.show()

