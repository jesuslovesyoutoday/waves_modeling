import numpy as np
import sympy as sp
from sympy import symbols
from sympy.functions import exp
from matplotlib import pyplot as plt

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

rmin = 0
rmax = 1.8
c = 1.5
a = 0.6
b = 1.2
f = 0
I = 100
T = 3
h = (rmax - rmin)/I
courant = 0.8
tau = courant * h / c
tsteps = int(T / tau)
times = np.linspace(0, T, tsteps)

R, A, B = symbols("r a b")
#U = exp((-4 * (2 * R - 1.8)**2)/((0.6)**2 - (2 * R - 1.8)**2))
U = exp(-((R+0.9)**2)/0.005)

x = np.array([rmin + (i - 0.5) * h for i in range(0, I + 1)])
u = np.zeros((tsteps, len(x)))
for i in range(len(x)):
        if ((x[i] > a) and (x[i] < b)):
            u[0, i] = U.subs({R:-x[i]})
plt.plot(x, u[0])
plt.show()
u1 = np.copy(u)

u1 = cartesian(u1, U, c, a, b, x, tau, tsteps, h, times, rmin, rmax)

"""u2 = np.copy(u)
u2 = cylindrical(u2, V, c, a, b, x, tau, tsteps, h, R)

u3 = np.copy(u)
u3 = spherical(u3, V, c, a, b, x, tau, tsteps, h, R)"""


import matplotlib.animation as animation
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
pl, = ax.plot(x, u1[0])
ax.set_ylim(-2, 2)
X = x

def update(frame, X, u):
    x = X
    y = u[frame]
    pl.set_data(x, y)
    return(pl)
    
ani = animation.FuncAnimation(fig, update, frames=tsteps, interval=30, fargs=[X, u1])
writer = animation.PillowWriter(fps=15,metadata=dict(artist='Me'),bitrate=1800)
ani.save('1d2.gif', writer=writer)

plt.close() 

"""cart = True
if cart:
    
    al = 1/3

    x1 = np.array([rmin + (i - 0.5) * h for i in range(0, I + 1)])
    x2 = np.array([rmin + (i - 0.5) * h*al for i in range(0, int(I/al) + 1)])
    x3 = np.array([rmin + (i - 0.5) * h*al*al for i in range(0, int(I/(al*al)) + 1)])

    u1h = np.zeros((tsteps, len(x1)))
    u1ha = np.zeros((int(tsteps/al), len(x2)))
    u1haa = np.zeros((int(tsteps/(al*al)), len(x3)))

    u1h = cartesian(u1h, U, c, a, b, x1, tau, tsteps, h, times, rmin, rmax)
    u1ha = cartesian(u1ha, U, c, a, b, x2, tau*al, int(tsteps/al), h*al, times, rmin, rmax)
    u1haa = cartesian(u1haa, U, c, a, b, x3, tau*al*al, int(tsteps/(al*al)), h*al*al, times, rmin, rmax)

    u1ha_  = np.zeros((tsteps, len(x1)))
    u1haa_ = np.zeros((tsteps, len(x1)))        
    
    for i in range(tsteps):
        for j in range(len(x1)):
            u1ha_[i, j]  = u1ha[i*3, j*3]
            u1haa_[i, j] = u1haa[i*9, j*9]
    plt.scatter(x1, u1h[2])
    plt.scatter(x1, u1ha_[2])
    plt.scatter(x1, u1haa_[2])
    plt.show()
    ratio_1d  = np.zeros(tsteps)
    ratioL_1d = np.zeros(tsteps)

    alpha = [1/(al*al) for i in range(tsteps)]

    for i in range(tsteps):
        ratio_1d[i]  = ratio(u1h[i], u1ha_[i], u1haa_[i])
        ratioL_1d[i] = ratioL(u1h[i], u1ha_[i], u1haa_[i], h, 1)

    plt.figure(figsize=(10,8))
    plt.plot(times, ratio_1d, label='Cnorm')
    plt.plot(times, ratioL_1d, label='Lnorm')
    plt.xlabel("Time", fontsize = 15)
    plt.ylabel("Ratio", fontsize = 15)
    plt.title("1D", fontsize=15)
    plt.plot(times, alpha, label='1/(alpha)^2')
    plt.legend(fontsize=15)
    plt.grid()
    plt.savefig("1d.pdf")
    plt.show()
"""
