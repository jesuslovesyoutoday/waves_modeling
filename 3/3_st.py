# import sympy as sp


# def __v0(x, a, b):
#     # x *= -1
#     return sp.exp(-4 * (2 * x - a - b)**2 / ((b - a) ** 2 - (2 * x - a - b)**2))


# x, a, b, d = sp.symbols('x a b d')

# res = __v0(x, a, b)

# d1 = sp.diff(res, x)

# print(d1)

# exit(0)


import numpy as np
import matplotlib.pyplot as plt

def _u(x, a, b):
    x *= -1
    if a < x < b:
        return np.exp(-4 * (2 * x - a - b)**2 / ((b - a) ** 2 - (2 * x - a - b)**2))
    else:
        return 0 * x

u = np.vectorize(_u, excluded={'a', 'b'})

def _dudx(x, a, b):
    x *= -1
    if a < x < b:
        # return (-4*(4*a + 4*b + 8*x)/((-a + b)**2 - (-a - b - 2*x)**2) -
        #        4*(-a - b - 2*x)**2*(4*a + 4*b + 8*x)/((-a + b)**2 -
        #        (-a - b - 2*x)**2)**2)*np.exp(-4*(-a - b - 2*x)**2/((-a + b)**2
        #        - (-a - b - 2*x)**2))
        return (-4*(-4*a - 4*b + 8*x)/((-a + b)**2 - (-a - b + 2*x)**2) - 4*(-4*a - 4*b + 8*x)*(-a - b + 2*x)**2/((-a + b)**2 - (-a - b + 2*x)**2)**2)*np.exp(-4*(-a - b + 2*x)**2/((-a + b)**2 - (-a - b + 2*x)**2))
    else:
        return 0*x

dudx = np.vectorize(_dudx, excluded={'a', 'b'})


def v0(r, d, a, b):
    return r**((1 - d) / 2) * u(-r, a, b)

def v1(r, d, ctau, a, b):
    return r**((1 - d) / 2) * u(ctau - r, a, b)

def dudr(x, r, d, a, b):
    return -(1 - d) / 2 * r ** (-(1+d)/2) * u(x, a, b) - r ** ((1-d)/2) * dudx(x, a, b)

def f(t, r, d, c, a, b):
    if d != 2:
        return 0*r
    else:
        return c**2 * (d-1)*(d-3)/4/r**((d+3)/2)*u(c*t - r, a, b)


def solve(grid, time, c, a, b, d):
    u = np.zeros((time.size, grid.size))

    h = grid[3] - grid[2]
    tau = time[1] - time[0]

    r_min = (grid[1] + grid[0]) / 2
    r_max = (grid[-1] + grid[-2]) / 2

    u[0, :] = v0(grid, d, a, b)
    u[1, :] = v1(grid, d, c*tau, a, b)

    for n in range(1, time.size - 1):
        u[n + 1, 1:-1] = 2 * u[n, 1:-1] - u[n - 1, 1:-1] + \
                        tau**2 * c**2 / h / grid[1:-1]**(d-1) * ((grid[1:-1] + h/2) ** (d-1) *
                        (u[n, 2:] - u[n, 1:-1]) / h - (grid[1:-1] - h/2) ** (d-1) *
                        (u[n, 1:-1] - u[n, :-2]) / h) + tau**2 * f(time[n], grid[1:-1], d, c, a, b)
        u[n+1, 0] = u[n+1, 1] + h * dudr(c*time[n+1] - r_min, r_min, d, a, b)
        u[n+1, -1] = u[n+1, -2] - h * dudr(c*time[n+1] - r_max, r_max, d, a, b)

    return u


def norm(vec):
    return np.max(np.abs(vec), axis=1)

def norm2(vec, h):
    return np.sqrt(np.sum(h*np.abs(vec)**2, axis=1))

def main():
    T = 1.5
    r_min = 0.1
    r_max = 1.9

    a = 0.7
    b = 1.3
    c = 1.6

    d = 3

    Nr = [165, 495]

    Cu = 0.01
    print(f'{Cu=}')
    sol = [1] * 2
    grid = [1] * 2
    time = [1] * 2
    rsol = [1] * 2

    Nr = np.array(Nr)

    h_1 = h = (r_max - r_min) / Nr[0]
    tau_1 = h_1*Cu/c
    Nt_1 = int(T / tau_1)

    Nt = Nt_1 * np.array([1, 3])

    for i in range(len(Nr)):
        h = (r_max - r_min) / (Nr[i])
        grid[i] = np.linspace(r_min, r_max + h, Nr[i] + 2)
        grid[i] -= h/2
        time[i] = np.linspace(0, T, Nt[i] + 1)
        sol[i] = solve(grid[i], time[i], c, a, b, d)

        gr, gt = np.meshgrid(grid[i], time[i])
        p = c*gt - gr
        rsol[i] = gr ** ((1 - d) / 2) * u(p, a, b)



    d1 = norm(sol[0][:, 1:-1] - rsol[0][:, 1:-1])[1:]
    d2 = norm(sol[1][:, 1:-1] - rsol[1][:, 1:-1])[3::3]

    d1_ = norm2(sol[0][:, 1:-1] - rsol[0][:, 1:-1], grid[0][1] - grid[0][0])[1:]
    d2_ = norm2(sol[1][:, 1:-1] - rsol[1][:, 1:-1], grid[1][1] - grid[1][0])[3::3]

    plt.figure(figsize=(9., 6), dpi=400)

    plt.plot(time[0][1:], d1/d2, label='C-norm')
    plt.plot(time[0][1:], d1_/d2_, label='$L_2$ - norm')
    plt.plot([0, T], [9.0, 9.0], 'g--', label='Reference = 9')
    plt.legend()
    
    plt.grid('on')
    plt.xlabel('T')
    plt.ylabel(r'$\frac{||u_h - u||}{||u_{\alpha h} - u||}$')
    plt.title(f'Cu: {Cu:.2f}, {d=}')

    #plt.savefig(f'theory{d}d.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()

