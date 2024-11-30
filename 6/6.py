import numpy as np
import sympy as sp
from sympy import symbols
from sympy.functions import exp

def integrate(y1, y2, y3):
    return (y1 + 4*y2 + y3)/6

def K(t, tau, c, e, filename, u1, u2, u3):

    l = 3
    
    coefs = np.loadtxt(filename)

    a1 = coefs[:, 0]
    a2 = coefs[:, 1]
    b1 = coefs[:, 2]
    b2 = coefs[:, 3]

    n = int(filename[4] + filename[5])

    a = np.empty([n,], dtype=np.complex128)
    a.real = a1
    a.imag = a2

    b = np.empty([n,], dtype=np.complex128)
    b.real = b1
    b.imag = b2

    a = a*l*l*c
    b = b*l*c

    time = t*tau
    
    k = 0
    
    for i in range(len(a)):
        e[t, i] = (np.exp(b[i]*time)*e[t-1, i] + 
                 + integrate(np.exp(b[i]*(0.5*tau)*u3), np.exp(0)*u2, np.exp(b[i]*(-0.5*tau)*u1)))
        k += a[i] * e[t, i]

    return (k)

def TBC(u, tau, h, c, t, N, e, filename, u1, u2, u3):
    res = h*(-u[t, N-1]+u[t-1, N-1]+u[t-1, N])
    res += tau*c*(-u[t-1, N]+u[t-1, N-1]+u[t, N-1])
    res -= c*tau*h*K(t, tau, c, e, filename, u1, u2, u3)
    res = res / (h+tau*c)
    return res

def o22(u, V, c, a, b, x, tau, tsteps, h, filename):
    dV = sp.diff(V, R)
    d2V = sp.diff(dV, R)
    d2u = np.zeros(len(x))

    n = int(filename[4] + filename[5])
    e = np.zeros((tsteps, n))   

    for i in range(len(x)):
        if ((x[i] > a) and (x[i] < b)):
            d2u[i] = c**2 * d2V.subs({R:x[i], A:a, B:b})
        
    u[1, :] = u[0, :] + (tau ** 2) * d2u / 2
    u[1, 0] = 0
    u1 = 0
    u2 = (u[0, len(x)-1] + u[0, len(x)-2] ) / 2
    u3 = (u[0, len(x)-1] + u[1, len(x)-2]) / 2
    u[1, len(x) - 1] = TBC(u, tau, h, c, 1, len(x)-1, e, filename, u1, u2, u3)

    for t in range(2, tsteps):
        for r in range(1, len(x) - 1):
            u[t, r] = (2*u[t-1, r] - u[t-2, r] + (tau*c)**2/(h**2) * 
                       (u[t-1, r+1] - 2 * u[t-1, r] + u[t-1, r-1]))
        u[t, 0] = 0
        u1 = (u[t-2, len(x)-2] + u[t-1, len(x)-1]) / 2
        u2 = (u[t-1, len(x)-1] + u[t-1, len(x)-2] ) / 2
        u3 = (u[t, len(x)-2] + u[t-1, len(x)-1]) / 2
        u[t, len(x) - 1] = TBC(u, tau, h, c, t, len(x)-1, e, filename, u1, u2, u3)

    return u

rmin = 0
rmax = 1.8
r2max = 2*rmax - rmin
c = 1.5
a = 0.6
b = 1.2
f = 0
I = 31
T = 3
h = (rmax - rmin)/(I-1)
I2 = int((r2max - rmin)/h)
courant = 0.8
tau = courant * h / c
tsteps = int(T / tau)
time = np.linspace(0, T, tsteps)

R, A, B = symbols("r a b")
V = exp((-4 * (2 * R - (A + B))**2)/((B - A)**2 - (2 * R - (A + B))**2))

x = np.array([rmin + (i - 0.5) * h for i in range(0, I + 1)])
u17 = np.zeros((tsteps, len(x)), dtype=np.float32)
u33 = np.zeros((tsteps, len(x)), dtype=np.float32)
u65 = np.zeros((tsteps, len(x)), dtype=np.float32)

x2 = np.array([rmin + (i - 0.5) * h for i in range(0, I2 + 1)])

u17_ = np.zeros((tsteps, len(x2)), dtype=np.float32)
u33_ = np.zeros((tsteps, len(x2)), dtype=np.float32)
u65_ = np.zeros((tsteps, len(x2)), dtype=np.float32)


for i in range(len(x)):
    if ((x[i] > a) and (x[i] < b)):
        u17[0, i] = V.subs({R:x[i], A:a, B:b})
        u33[0, i] = V.subs({R:x[i], A:a, B:b})
        u65[0, i] = V.subs({R:x[i], A:a, B:b})

for i in range(len(x2)):
    if ((x2[i] > a) and (x2[i] < b)):
        u17_[0, i] = V.subs({R:x2[i], A:a, B:b})
        u33_[0, i] = V.subs({R:x2[i], A:a, B:b})
        u65_[0, i] = V.subs({R:x2[i], A:a, B:b})

u17 = o22(u17, V, c, a, b, x, tau, tsteps, h, "coef17.dat")
u33 = o22(u33, V, c, a, b, x, tau, tsteps, h, "coef33.dat")
u65 = o22(u65, V, c, a, b, x, tau, tsteps, h, "coef65.dat")

u17_ = o22(u17_, V, c, a, b, x2, tau, tsteps, h, "coef17.dat")
u33_ = o22(u33_, V, c, a, b, x2, tau, tsteps, h, "coef33.dat")
u65_ = o22(u65_, V, c, a, b, x2, tau, tsteps, h, "coef65.dat")

"""import matplotlib.animation as animation
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
pl, = ax.plot(x, u17[0])
ax.set_ylim(-2, 2)
X = x

def update(frame, X, u17):
    x = X
    y = u17[frame]
    pl.set_data(x, y)
    return(pl)
    
ani = animation.FuncAnimation(fig, update, frames=tsteps, interval=30, fargs=[X, u17])
writer = animation.PillowWriter(fps=15,metadata=dict(artist='Me'),bitrate=1800)
ani.save('tbc.gif', writer=writer)

plt.close()"""

import matplotlib.pyplot as plt

t0 = 90

du17 = u17_[t0, 0:len(x)] - u17[t0]
du33 = u33_[t0, 0:len(x)] - u33[t0]
du65 = u65_[t0, 0:len(x)] - u65[t0]

"""fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(x, u17[t0])
axs[1].plot(x, u33[t0])
axs[2].plot(x, u65[t0])
plt.show()"""

delta = np.zeros(tsteps)
for i in range(tsteps):
    delta[i] = np.max(np.abs(u17_[i, 0:len(x)] - u17[i]))
plt.plot(time, delta)
plt.show()
