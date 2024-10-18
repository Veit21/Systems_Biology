import numpy as np
import matplotlib.pyplot as plt

CARRYING_CAPACITY = 100
BIRTH_RATE = 1
N_0 = 1
DELTA_H = [0.01, 0.05, 0.1, 1, 2, 5]
T_0 = 0
T_END = 15
TIME_ARRAY = np.linspace(T_0, T_END, 100)


def n_prime(n_t):
    """ Differential equation for logistic growth. """
    return BIRTH_RATE * n_t * (1 - n_t/CARRYING_CAPACITY)


def function_n(t):
    """ Analytical solution of the ODE. """
    n_t = (N_0 * CARRYING_CAPACITY * np.exp(BIRTH_RATE * t)) / (CARRYING_CAPACITY + N_0 * (np.exp(BIRTH_RATE * t) - 1))
    return n_t


def euler_method(function, t_0, n_0, d_h, t_end=T_END):
    """ Euler method - a method to numerically solve an ODE. """
    n = n_0
    t_steps = np.arange(t_0, t_end + d_h, d_h)
    n_approx = [n]
    for t in t_steps:
        n += d_h * function(t, n)
        n_approx.append(n)
    return t_steps, n_approx[:-1]


def runge_kutta(function, t_0, n_0, d_h, t_end=T_END):
    """ Runge-Kutta 4 method - another method to numerically solve an ODE. """
    n = n_0
    steps = np.arange(t_0, t_end + d_h, d_h)
    n_approx = [n]
    for t in steps:
        k1 = function(t, n)
        k2 = function(t + d_h/2, n + k1 * d_h/2)
        k3 = function(t + d_h/2, n + k2 * d_h/2)
        k4 = function(t + d_h, n + k3 * d_h)
        k_array = np.array([k1, k2, k3, k4])
        weights = np.array([1/6, 1/3, 1/3, 1/6])
        n += d_h * np.dot(weights, k_array)
        n_approx.append(n)
    return steps, n_approx[:-1]


# calculate analytical solution of the ODE
N_t = function_n(TIME_ARRAY)

# plot all graphs
fig1, ax1 = plt.subplots(nrows=2, ncols=3)
fig2, ax2 = plt.subplots(nrows=2, ncols=3)
run_index = 0
for i, row in enumerate(ax1):
    for j, col in enumerate(row):
        euler_steps, N_t_euler = euler_method(lambda t, n: n_prime(n), T_0, N_0, DELTA_H[run_index])
        rk4_steps, N_t_rk4 = runge_kutta(lambda t, n: n_prime(n), T_0, N_0, DELTA_H[run_index])
        euler_ek4_diff = abs(np.asarray(N_t_euler) - np.asarray(N_t_rk4))
        col.plot(TIME_ARRAY, N_t, c='k', linewidth='0.5', label='Analytic')
        col.scatter(euler_steps, N_t_euler, s=4, color='green', label='Euler')
        col.scatter(rk4_steps, N_t_rk4, s=4, color='m', label='Runge-Kutta')
        col.legend()
        col.set_title('Δt = {}'.format(DELTA_H[run_index]))
        col.set_xlabel('t')
        col.set_ylabel('N(t)')
        ax2[i][j].plot(euler_steps, euler_ek4_diff, c='m', linewidth='1')
        ax2[i][j].set_title('Δt = {}'.format(DELTA_H[run_index]))
        ax2[i][j].set_xlabel('t')
        ax2[i][j].set_ylabel('ΔN(t)')
        run_index += 1

plt.show()
