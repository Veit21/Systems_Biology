
# imports
import sys
# sys.path.remove('/Users/veithucke/.local/lib/python3.10/site-packages')     # TODO: Had some problems with dependencies and path variables. Remove again!
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from scipy.integrate import odeint

axis_color = 'white'
ts = np.linspace(0, 30, 1000)   # time grid


# solve the system dy/dt = f(y, t)
def modelSZR(y, t, a, Ni):
    Si, Zi, Ri = y

    # the model equations
    f0 = - Si * Zi / Ni
    f1 = (1 - a) * Si * Zi / Ni
    f2 = a * Si * Zi / Ni
    return [f0, f1, f2]


def modelSIR(y, t, mu, Ni):
    Si, Ii, Ri = y

    # the model equations
    f0 = - Si * Ii / Ni
    f1 = ((Si / Ni) - mu) * Ii
    f2 = mu * Ii
    return [f0, f1, f2]


# initial conditions
S0 = 199                        # initial susceptible population
Z0 = 1                          # initial zombie population
I0 = Z0                         # initial infected population for SIR model
R0 = 0                          # initial death population
N = S0 + Z0 + R0                # total population
y0 = [S0, Z0, R0]               # initial condition vector
a0 = 0.6                        # dimensionless virulence (a = k/b)
mu = 0.6                        # dimensionless "infectivity" (inverse of R0 for closer analogy to a of SZR)


# solve the DEs of SIR model
solSIR = odeint(modelSIR, y0, ts, args=(mu, N))
S_sir = solSIR[:, 0]
I_sir = solSIR[:, 1]
R_sir = solSIR[:, 2]


# solve the DEs of SZR model
solSZR = odeint(modelSZR, y0, ts, args=(a0, N))
S_szr = solSZR[:, 0]
Z_szr = solSZR[:, 1]
R_szr = solSZR[:, 2]


# plot results
fig = plt.figure()
[line1] = plt.plot(ts, S_szr, 'b', lw=2, label='Susceptible')
[line2] = plt.plot(ts, Z_szr, 'r', lw=2, label='Zombies')
[line3] = plt.plot(ts, R_szr, 'k', lw=2, label='Removed')
[line4] = plt.plot(ts, S_sir, 'lightsteelblue', lw=1.5, label='Susceptible')
[line5] = plt.plot(ts, I_sir, 'lightcoral', lw=1.5, label='Infected')
[line6] = plt.plot(ts, R_sir, 'darkgray', lw=1.5, label='Recovered')
plt.xlabel('t')
plt.ylabel('Population')
plt.legend(loc="upper right",bbox_to_anchor=(1.1, 1.1), fancybox=True)
plt.grid(True, ls='-.')
plt.subplots_adjust(bottom=0.25)

# Define an axes area and draw a slider in it
a_slider_ax = fig.add_axes([0.3, 0.1, 0.4, 0.04], facecolor=axis_color)
z_slider_ax = fig.add_axes([0.3, 0.01, 0.4, 0.04], facecolor=axis_color)
a_slider = Slider(a_slider_ax, '⍺ & μ', 0.1, 1.5, valinit=a0)
z_slider = Slider(z_slider_ax, 'Z0', 1.0, 100.0, valinit=Z0)

# Checkbutton widget
labels = ['SZR', 'SIR']
activated = [True, True]
axCheckButtons = plt.axes([0.1, 0.1, 0.1, 0.1])
checkBox = CheckButtons(axCheckButtons, labels, activated)

# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.1, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')


# Define action for pressing the reset button
def reset_button_on_clicked(mouse_event):
    a_slider.reset()
    z_slider.reset()


# Define action for pressing check buttons
def set_visible(label):
    if label == "SZR":
        line1.set_visible(not line1.get_visible())
        line2.set_visible(not line2.get_visible())
        line3.set_visible(not line3.get_visible())
        plt.draw()
    else:
        line4.set_visible(not line4.get_visible())
        line5.set_visible(not line5.get_visible())
        line6.set_visible(not line6.get_visible())
        plt.draw()


# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    N_new = 200
    Z_new = z_slider.val
    sol_szr = odeint(modelSZR, [N_new - Z_new, Z_new, R0], ts, args=(a_slider.val, N_new))
    S_szr = sol_szr[:, 0]
    Z_szr = sol_szr[:, 1]
    R_szr = sol_szr[:, 2]

    sol_sir = odeint(modelSIR, [N_new - Z_new, Z_new, R0], ts, args=(a_slider.val, N_new))
    S_sir = sol_sir[:, 0]
    I_sir = sol_sir[:, 1]
    R_sir = sol_sir[:, 2]

    line1.set_ydata(S_szr)
    line2.set_ydata(Z_szr)
    line3.set_ydata(R_szr)
    line4.set_ydata(S_sir)
    line5.set_ydata(I_sir)
    line6.set_ydata(R_sir)
    fig.canvas.draw_idle()


a_slider.on_changed(sliders_on_changed)
z_slider.on_changed(sliders_on_changed)
reset_button.on_clicked(reset_button_on_clicked)
checkBox.on_clicked(set_visible)
plt.show()