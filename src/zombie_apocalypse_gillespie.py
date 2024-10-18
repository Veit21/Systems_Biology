
# imports
import sys
# sys.path.remove('/Users/veithucke/.local/lib/python3.10/site-packages')     # TODO: Had some problems with dependencies and path variables. Remove again!
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# initial conditions
S0 = [199]                        # initial susceptible population
Z0 = [1]                          # initial zombie population
R0 = [0]                          # initial death population
t0 = [0]                          # initial time t0

k = 0.06                          # kill parameter, k/b = 0.6
b = 0.1                           # bite parameter
t_end = 30                        # endpoint of the simulation


# actual simulation loop
def run_simulation(S, Z, R, t, t_end, kill_rate, bite_rate):
    while t[-1] < t_end:

        # 1st reaction: (S,Z) --{b*S*Z}--> (Z,Z)
        # 2nd reaction: (S,Z) --{k*S*Z}--> (S,R)

        propensities = [bite_rate * S[-1] * Z[-1], kill_rate * S[-1] * Z[-1]]   # always taking the last value of S and Z from the array
        propensity_sum = np.sum(propensities)

        if propensity_sum == 0:
            break

        r1 = np.random.uniform(0, 1)                 # necessary to determine how long it takes for the next time step
        r2 = np.random.uniform(0, 1)                 # necessary to determine which reaction happens next
        tau = (1/propensity_sum) * np.log(1/r1)      # time until next time step (sojourn time)
        t.append(t[-1] + tau)                        # updating the time array

        # updating the populations
        if r2 * propensity_sum <= propensities[0]:
            S.append(S[-1] - 1)
            Z.append(Z[-1] + 1)
            R.append(R[-1])
        else:
            S.append(S[-1])
            Z.append(Z[-1] - 1)
            R.append(R[-1] + 1)
    return S, Z, R, t


# run simulation
S, Z, R, t = run_simulation(S0, Z0, R0, t0, t_end, k, b)


# plot everything
fig = plt.figure()
S_plot, = plt.plot(t, S, 'b', lw=2, label='Susceptible')
Z_plot, = plt.plot(t, Z, 'r', lw=2, label='Zombie')
R_plot, = plt.plot(t, R, 'k', lw=2, label='Removed')
plt.xlabel('t')
plt.ylabel('Population')
plt.legend(handles=[S_plot, Z_plot, R_plot])
plt.grid(True, ls='-.')

# Add a button for resetting the parameters
# next_button_ax = fig.add_axes([0.88, 0.02, 0.1, 0.04])
# next_button = Button(next_button_ax, 'Next', hovercolor='0.975')


# Add function to the Next button
def next_simulation(mouse_event):
    s0 = [199]
    z0 = [1]
    r0 = [0]
    t_null = [0]
    S_new, Z_new, R_new, t_new = run_simulation(s0, z0, r0, t_null, t_end, k, b)
    S_plot.set_data(t_new, S_new)
    Z_plot.set_data(t_new, Z_new)
    R_plot.set_data(t_new, R_new)
    fig.canvas.draw()


# next_button.on_clicked(next_simulation)
plt.show()