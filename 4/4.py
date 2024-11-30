import numpy as np
import sympy as sp
from sympy import symbols
from sympy.functions import exp
import matplotlib.pyplot as plt


def O42(U, c, a, b, x, tsteps, h, time):

    u = np.zeros((len(time), len(x)))

    coeff = [-5/4, 4/3, -1/12]

    tau = time[1] - time[0]

    for i in range(len(x)):
        if ((x[i] > a) and (x[i] < b)):
            u[0, i] = U.subs({R:-x[i]})
            u[1, i] = U.subs({R:(c*tau-x[i])})    

    for t in range(2, tsteps):
        for r in range(2, len(x) - 2):
            summ = 0
            for k in range(3):
                summ += coeff[k] * (u[t-1, r-k] + u[t-1, r+k])
            u[t, r] = (c*tau/h)**2 * summ + 2*u[t-1, r] - u[t-2, r]
    return u            

def O62(U, c, a, b, x, tsteps, h, time):

    u = np.zeros((len(time), len(x)))

    coeff = [-49/36, 3/2, -3/20, 1/90]

    tau = time[1] - time[0]

    for i in range(len(x)):
        if ((x[i] > a) and (x[i] < b)):
            u[0, i] = U.subs({R:-x[i]})
            u[1, i] = U.subs({R:(c*tau-x[i])})

    for t in range(2, tsteps):
        for r in range(3, len(x) - 3):
            summ = 0
            for k in range(4):
                summ += coeff[k] * (u[t-1, r-k] + u[t-1, r+k])
            u[t, r] = (c*tau/h)**2 * summ + 2*u[t-1, r] - u[t-2, r]
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
I = 100
T = 0.3
I0 = 50
al = 3

R, A, B = symbols("r a b")
V = exp((-4 * (2 * R - (A + B))**2)/((B - A)**2 - (2 * R - (A + B))**2))
V = exp(-((R+0.9)**2)/0.005)

I = np.array([I0, I0*al, I0*al*al])
courant = 0.5

h = np.array([])
h = (rmax - rmin)/I

o42 = False
o62 = True

tau1 = h[0] * courant / c
tstep1 = int(T/tau1)
if (o42):
    tsteps = tstep1 * np.array([1, 9, 81])
elif (o62):
    tsteps = tstep1 * np.array([1, 27, 729])

x = [0] * 3
time = [0] * 3
u = [0] * 3


for i in range(3):
    x[i] = np.linspace(rmin, rmax + h[i], I[i] + 2)
    x[i] -= h[i]/2
    time[i] = np.linspace(0, T, tsteps[i] + 1)
    if (o42):
        u[i] = O42(V, c, a, b, x[i], tsteps[i], h[i], time[i])
        title = 'o42'
    elif (o62):
        u[i] = O62(V, c, a, b, x[i], tsteps[i], h[i], time[i])
        title = 'o62'

t1 = 2
t2 = 5

if (o42):
    c1 = cnorm(u[1][::9, t1:-1:3] - u[0][:, 1:-1])[1:]
    c2 = cnorm(u[2][::81, t2:-1:9] - u[1][::9, t1:-1:3])[1:]

    l1 = lnorm(u[1][::9, t1:-1:3] - u[0][:, 1:-1], h[0])[1:]
    l2 = lnorm(u[2][::81, t2:-1:9] - u[1][::9, t1:-1:3], h[0])[1:]

elif(o62):
    c1 = cnorm(u[1][::27, t1:-1:3] - u[0][:, 1:-1])[1:]
    c2 = cnorm(u[2][::729, t2:-1:9] - u[1][::27, t1:-1:3])[1:]

    l1 = lnorm(u[1][::27, t1:-1:3] - u[0][:, 1:-1], h[0])[1:]
    l2 = lnorm(u[2][::729, t2:-1:9] - u[1][::27, t1:-1:3], h[0])[1:]

plt.figure(figsize=(10,8))
plt.plot(time[0][1:], c1/c2, label='Cnorm')
plt.plot(time[0][1:], l1/l2, label='Lnorm')
plt.xlabel("Time", fontsize = 15)
plt.ylabel("Ratio", fontsize = 15)
plt.title(title, fontsize=15)
if (o42):
    plt.plot([0, T], [81.0, 81.0], label='1/(alpha)^2')
elif(o62):
    plt.plot([0, T], [729.0, 729.0], label='1/(alpha)^2')
plt.legend(fontsize=15)
plt.grid()
plt.savefig(title + ".pdf")
plt.show()

