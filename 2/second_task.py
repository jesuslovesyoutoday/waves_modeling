import numpy as np
import sympy as sp
from sympy import symbols
from math import pow
from sympy.functions import exp

import matplotlib.pyplot as plt

def cartesian(u, V, c, a, b, x, tau, tsteps, h, times):
    """dV = sp.diff(V, R)
    d2V = sp.diff(dV, R)
    d2u = np.zeros(len(x))
    
    for i in range(len(x)):
        if ((x[i] > a) and (x[i] < b)):
            d2u[i] = c**2 * d2V.subs({R:x[i], A:a, B:b})
        
    u[1, :] = u[0, :] + (tau ** 2) * d2u / 2
    u[1, 0] = u[1, 1]
    u[1, len(x) - 1] = u[1, len(x) - 2]"""

    for t in range(2, tsteps):
        for r in range(1, len(x) - 2):
            """u[t, r] = (2*u[t-1, r] - u[t-2, r] + (tau*c)**2/(h**2) * 
                       (u[t-1, r+1] - 2 * u[t-1, r] + u[t-1, r-1]))
        u[t, 0] = u[t, 1]
        u[t, len(x) - 1] = u[t, len(x) - 2]"""
        u[t, r] = times[t] * x[r]
        
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
I = 210
T = 3
h = (rmax - rmin)/I
courant = 0.09
tau = courant * h / c
tsteps = int(T / tau)

R, A, B = symbols("r a b")
V = exp((-4 * (2 * R - (A + B))**2)/((B - A)**2 - (2 * R - (A + B))**2))

x = np.array([rmin + (i - 0.5) * h for i in range(0, I + 1)])
u = np.zeros((tsteps, len(x)))
times = np.linspace(0, T, tsteps)

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

    tsteps2 = int(tsteps/al)
    tsteps3 = int(tsteps/(al*al))

    times = np.linspace(0, T, tsteps)
    times2 = np.linspace(0, T, tsteps2)  
    times3 = np.linspace(0, T, tsteps3)

    u1h = np.zeros((tsteps, len(x1)))
    u1ha = np.zeros((tsteps2, len(x2)))
    u1haa = np.zeros((tsteps3, len(x3)))

    for i in range(len(x1)):
        if ((x1[i] > a) and (x1[i] < b)):
            u1h[0, i] = V.subs({R:x1[i], A:a, B:b})

    for i in range(len(x2)):
        if ((x2[i] > a) and (x2[i] < b)):
            u1ha[0, i] = V.subs({R:x2[i], A:a, B:b})

    for i in range(len(x3)):
        if ((x3[i] > a) and (x3[i] < b)):
            u1haa[0, i] = V.subs({R:x3[i], A:a, B:b})

    tau2 = T/tsteps2 + 1
    tau3 = T/tsteps3 + 1
    
    u1h = cartesian(u1h, V, c, a, b, x1, tau, tsteps, h, times)
    u1ha = cartesian(u1ha, V, c, a, b, x2, tau2, tsteps2, h*al, times2)
    u1haa = cartesian(u1haa, V, c, a, b, x3, tau3, tsteps3, h*al*al, times3)

    u1ha_  = np.zeros((tsteps, len(x1)))
    u1haa_ = np.zeros((tsteps, len(x1)))        
    
    for i in range(tsteps):
        for j in range(len(x1)):
            u1ha_[i, j]  = u1ha[i*3, j*3]
            u1haa_[i, j] = u1haa[i*9, j*9]
    """plt.scatter(x1, u1h[2])
    plt.scatter(x1, u1ha_[2])
    plt.scatter(x1, u1haa_[2])
    plt.show()"""
    ratio_1d  = np.zeros(tsteps)
    ratioL_1d = np.zeros(tsteps)

    alpha = [1/(al*al) for i in range(tsteps)]

    """plt.plot(x, u1h[20] - u1ha_[20], label='1')
    plt.plot(x, u1ha_[20] - u1haa_[20], label='2')
    plt.legend()
    plt.show()"""
    for i in range(tsteps):
        ratio_1d[i]  = ratio(u1h[i], u1ha_[i], u1haa_[i])
        ratioL_1d[i] = ratioL(u1h[i], u1ha_[i], u1haa_[i], h, 1)

    """plt.figure(figsize=(10,8))
    plt.plot(times, ratio_1d, label='Cnorm')
    plt.plot(times, ratioL_1d, label='Lnorm')
    plt.xlabel("Time", fontsize = 15)
    plt.ylabel("Ratio", fontsize = 15)
    plt.title("1D", fontsize=15)
    plt.plot(times, alpha, label='1/(alpha)^2')
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig("1d.pdf")
    plt.show()"""
    check1 = [u1h[i] - u1ha_[i] for i in range (tsteps)]
    check2 = [u1ha_[i] - u1haa_[i] for i in range (tsteps)]
    print(check1, check2)

"""if cyl:
    
    al = 1/3

    x1 = np.array([rmin + (i - 0.5) * h for i in range(0, I + 1)])
    x2 = np.array([rmin + (i - 0.5) * h*al for i in range(0, int(I/al) + 1)])
    x3 = np.array([rmin + (i - 0.5) * h*al*al for i in range(0, int(I/(al*al)) + 1)])

    u2h = np.zeros((tsteps, len(x1)))
    u2ha = np.zeros((int(tsteps/al), len(x2)))
    u2haa = np.zeros((int(tsteps/(al*al)), len(x3)))

    for i in range(len(x1)):
        if ((x1[i] > a) and (x1[i] < b)):
            u2h[0, i] = V.subs({R:x1[i], A:a, B:b})

    for i in range(len(x2)):
        if ((x2[i] > a) and (x2[i] < b)):
            u2ha[0, i] = V.subs({R:x2[i], A:a, B:b})

    for i in range(len(x3)):
        if ((x3[i] > a) and (x3[i] < b)):
            u2haa[0, i] = V.subs({R:x3[i], A:a, B:b})

    
    u2h = cartesian(u2h, V, c, a, b, x1, tau, tsteps, h)
    u2ha = cartesian(u2ha, V, c, a, b, x2, tau*al, int(tsteps/al), h*al)
    u2haa = cartesian(u2haa, V, c, a, b, x3, tau*al*al, int(tsteps/(al*al)), h*al*al)

    u2ha_  = np.zeros((tsteps, len(x1)))
    u2haa_ = np.zeros((tsteps, len(x1)))        
    
    for i in range(tsteps):
        for j in range(len(x1)):
            u2ha_[i, j]  = u2ha[i*3, j*3]
            u2haa_[i, j] = u2haa[i*9, j*9]

    ratio_2d  = np.zeros(tsteps)
    ratioL_2d = np.zeros(tsteps)

    alpha = [1/(al*al) for i in range(tsteps)]

    for i in range(tsteps):
        ratio_2d[i]  = ratio(u2h[i], u2ha_[i], u2haa_[i])
        ratioL_2d[i] = ratioL(u2h[i], u2ha_[i], u2haa_[i], h, 2)

    plt.figure(figsize=(10,8))
    plt.plot(times, ratio_2d, label='Cnorm')
    plt.plot(times, ratioL_2d, label='Lnorm')
    plt.xlabel("Time", fontsize = 15)
    plt.ylabel("Ratio", fontsize = 15)
    plt.title("1D", fontsize=15)
    plt.plot(times, alpha, label='1/(alpha)^2')
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig("2d.pdf")
    plt.show()

if spher:
    
    al = 1/3

    x1 = np.array([rmin + (i - 0.5) * h for i in range(0, I + 1)])
    x2 = np.array([rmin + (i - 0.5) * h*al for i in range(0, int(I/al) + 1)])
    x3 = np.array([rmin + (i - 0.5) * h*al*al for i in range(0, int(I/(al*al)) + 1)])

    u3h = np.zeros((tsteps, len(x1)))
    u3ha = np.zeros((int(tsteps/al), len(x2)))
    u3haa = np.zeros((int(tsteps/(al*al)), len(x3)))

    for i in range(len(x1)):
        if ((x1[i] > a) and (x1[i] < b)):
            u3h[0, i] = V.subs({R:x1[i], A:a, B:b})

    for i in range(len(x2)):
        if ((x2[i] > a) and (x2[i] < b)):
            u3ha[0, i] = V.subs({R:x2[i], A:a, B:b})

    for i in range(len(x3)):
        if ((x3[i] > a) and (x3[i] < b)):
            u3haa[0, i] = V.subs({R:x3[i], A:a, B:b})

    
    u3h = cartesian(u3h, V, c, a, b, x1, tau, tsteps, h)
    u3ha = cartesian(u3ha, V, c, a, b, x2, tau*al, int(tsteps/al), h*al)
    u3haa = cartesian(u3haa, V, c, a, b, x3, tau*al*al, int(tsteps/(al*al)), h*al*al)

    u3ha_  = np.zeros((tsteps, len(x1)))
    u3haa_ = np.zeros((tsteps, len(x1)))        
    
    for i in range(tsteps):
        for j in range(len(x1)):
            u3ha_[i, j]  = u3ha[i*3, j*3]
            u3haa_[i, j] = u3haa[i*9, j*9]

    ratio_3d  = np.zeros(tsteps)
    ratioL_3d = np.zeros(tsteps)

    alpha = [1/(al*al) for i in range(tsteps)]

    for i in range(tsteps):
        ratio_3d[i]  = ratio(u3h[i], u3ha_[i], u3haa_[i])
        ratioL_3d[i] = ratioL(u3h[i], u3ha_[i], u3haa_[i], h, 3)

    plt.figure(figsize=(10,8))
    plt.plot(times, ratio_3d, label='Cnorm')
    plt.plot(times, ratioL_3d, label='Lnorm')
    plt.xlabel("Time", fontsize = 15)
    plt.ylabel("Ratio", fontsize = 15)
    plt.title("1D", fontsize=15)
    plt.plot(times, alpha, label='1/(alpha)^2')
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig("1d.pdf")
    plt.show()
        """

