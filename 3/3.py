import numpy as np
import sympy as sp
from sympy import symbols
from math import pow
from sympy.functions import exp

import matplotlib.pyplot as plt

def cartesian(u, U, c, a, b, x, tau, tsteps, h, times, rmin, rmax):

    for i in range(len(x)):
        if ((x[i] > a) and (x[i] < b)):
            u[0, i] = U.subs({R:-x[i]})
            u[1, i] = U.subs({R:(c*tau-x[i])})    
    dU = sp.diff(U, R)
    du = -dU

    for t in range(2, tsteps):
        for r in range(1, len(x) - 2):
            u[t, r] = (2*u[t-1, r] - u[t-2, r] + (tau*c)**2/(h**2) * 
                       (u[t-1, r+1] - 2 * u[t-1, r] + u[t-1, r-1]))
        u[t, 0] = u[t, 1] + h*du.subs({R:(c*times[t]-rmin)})/2
        u[t, len(x) - 1] = u[t, len(x) - 2] + h*du.subs({R:(c*times[t]-rmax)})/2
        #plt.plot(x, u[t])
        #plt.ylim(bottom=-2)
        #plt.ylim(top=2)
        #plt.show()
    return u

def cylindrical(u, V, c, a, b, x, tau, tsteps, h, R):
    dV = sp.diff(V, R)
    d2u = np.zeros(len(x))
    drdV = sp.diff(dV*R, R)
    
    for i in range(len(x)):
        if ((x[i] > a) and (x[i] < b)):
            d2u[i] = c**2 * drdV.subs({R:x[i], A:a, B:b}) / x[i]
    
    u[1, :] = u[0, :] + (tau ** 2) * d2u / 2
    u[1, 0] = u[1, 1]
    u[1, len(x) - 1] = u[1, len(x) - 2]
    
    for t in range(2, tsteps):
        for r in range(1, len(x) - 2):
            u[t, r] = (2*u[t-1, r] - u[t-2, r] + (tau*c)**2/(h**2)/x[r] * 
                      ((x[r] + h/2) * (u[t-1, r+1] - u[t-1, r]) -
                       (x[r] - h/2) * (u[t-1, r] - u[t-1, r-1])))
        u[t, 0] = u[t, 1]
        u[t, len(x) - 1] = u[t, len(x) - 2]
        
    return u

def spherical(u, V, c, a, b, x, tau, tsteps, h, R):
    dV = sp.diff(V, R)
    d2u = np.zeros(len(x))
    drdV = sp.diff(dV*R*R, R)
    
    for i in range(len(x)):
        if ((x[i] > a) and (x[i] < b)):
            d2u[i] = c**2 * drdV.subs({R:x[i], A:a, B:b}) / x[i] / x[i]
    
    u[1, :] = u[0, :] + (tau ** 2) * d2u / 2
    u[1, 0] = u[1, 1]
    u[1, len(x) - 1] = u[1, len(x) - 2]
    
    for t in range(2, tsteps):
        for r in range(1, len(x) - 2):
            u[t, r] = (2*u[t-1, r] - u[t-2, r] + (tau*c)**2/(h**2)/x[r]/x[r] * 
                      (((x[r] + h/2)**2) * (u[t-1, r+1] - u[t-1, r]) -
                       ((x[r] - h/2)**2) * (u[t-1, r] - u[t-1, r-1])))
        u[t, 0] = u[t, 1]
        u[t, len(x) - 1] = u[t, len(x) - 2]
        
    return u

def cnorm(u):
    return np.max(np.abs(u))

def lnorm(u, h, d):
    return np.sqrt(np.sum([(abs(i)**2) * pow(h, d) for i in u]))

def ratio(uh, uha, uhaa):
    return ((cnorm(uh - uha))/(cnorm(uha - uhaa)))

def ratioL(uh, uha, uhaa, h, d):
    return ((lnorm(uh - uha, h, d))/(lnorm(uha - uhaa, h, d)))

rmin = 0
rmax = 1.8
c = 1.5
a = 0.6
b = 1.2
f = 0
I = 100
T = 3
h = (rmax - rmin)/I
courant = 0.1
tau = courant * h / c
tsteps = int(T / tau)

R, A, B = symbols("r a b")
V = exp((-4 * (2 * R - (A + B))**2)/((B - A)**2 - (2 * R - (A + B))**2))
V = exp(-((R+0.9)**2)/0.005)

x = np.array([rmin + (i - 0.5) * h for i in range(0, I + 1)])
u = np.zeros((tsteps, len(x)))


for i in range(len(x)):
    if ((x[i] > a) and (x[i] < b)):
        u[0, i] = V.subs({R:x[i], A:a, B:b})

cart = True
cyl = False
spher = False
al = 1/3

if cart:
    
    al = 1/3

    x1 = np.array([rmin + (i - 0.5) * h for i in range(0, I + 1)])
    x2 = np.array([rmin + (i - 0.5) * h*al for i in range(0, int(I/al) + 1)])
    x3 = np.array([rmin + (i - 0.5) * h*al*al for i in range(0, int(I/(al*al)) + 1)])

    u1h = np.zeros((tsteps, len(x1)))
    u1ha = np.zeros((int(tsteps/al), len(x2)))
    u1haa = np.zeros((int(tsteps/(al*al)), len(x3)))

    times1 = np.linspace(0, T, tsteps)
    times2 = np.linspace(0, T, tsteps*3)
    times3 = np.linspace(0, T, tsteps*9)

    u1h = cartesian(u1h, V, c, a, b, x1, tau, tsteps, h, times1, rmin, rmax)
    u1ha = cartesian(u1ha, V, c, a, b, x2, tau*al, int(tsteps/al), h*al, times2, rmin, rmax)
    u1haa = cartesian(u1haa, V, c, a, b, x3, tau*al*al, int(tsteps/(al*al)), h*al*al, times3, rmin, rmax)

    u1ha_  = np.zeros((tsteps, len(x1)))
    u1haa_ = np.zeros((tsteps, len(x1)))        
    
    for i in range(tsteps):
        for j in range(len(x1)):
            u1ha_[i, j]  = u1ha[i*3, j*3]
            u1haa_[i, j] = u1haa[i*9, j*9]
    plt.plot(x1, u1h[15])
    plt.plot(x1, u1ha_[15])
    plt.plot(x1, u1haa_[15])
    plt.show()
    plt.legend()
    ratio_1d  = np.zeros(tsteps)
    ratioL_1d = np.zeros(tsteps)

    alpha = [1/(al*al) for i in range(tsteps)]

    for i in range(tsteps):
        ratio_1d[i]  = ratio(u1h[i], u1ha_[i], u1haa_[i])
        ratioL_1d[i] = ratioL(u1h[i], u1ha_[i], u1haa_[i], h, 1)

    plt.figure(figsize=(10,8))
    plt.plot(times1, ratio_1d, label='Cnorm')
    plt.plot(times1, ratioL_1d, label='Lnorm')
    plt.xlabel("Time", fontsize = 15)
    plt.ylabel("Ratio", fontsize = 15)
    plt.title("1D", fontsize=15)
    plt.plot(times1, alpha, label='1/(alpha)^2')
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig("1d.pdf")
    plt.show()

