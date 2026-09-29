from pyBondGraph import BondGraph, SourceEffort, Inductor, Resistor, OneJunction, Gyrator

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

bond_graph = BondGraph()

voltage_source = SourceEffort("V", "U_A(t)")
junction_elec = OneJunction("J1_1")
inductor = Inductor("I_elec", "L_A", numeric_value=15e-6)
resistor = Resistor("R_elec", "R_A", numeric_value=4)
gyrator = Gyrator("G1", "K_t", numeric_value=9.54e-3)
junction_mech = OneJunction("J1_2")
bearing = Resistor("R_mech", "R_B", numeric_value=1e-6)
inertia = Inductor("I_mech", "J", numeric_value=1e-6)

bond_graph.connect(voltage_source, junction_elec)
bond_graph.connect(junction_elec, resistor)
bond_graph.connect(junction_elec, inductor)
bond_graph.connect(junction_elec, gyrator)
bond_graph.connect(gyrator, junction_mech)
bond_graph.connect(junction_mech, bearing)
bond_graph.connect(junction_mech, inertia)

bond_graph.assign_causality()

print(bond_graph.get_solution_equations())

A, B, C, D, x, n_states, n_inputs, n_outputs = bond_graph.get_state_space()


# Print results
print("\nState vector x:")
sp.pprint(x)
print("Matrix A:")
sp.pprint(A)
print("\nMatrix B:")
sp.pprint(B)
print("\nMatrix C:")
sp.pprint(C)
print("\nMatrix D:")
sp.pprint(D)

fig, ax = bond_graph.plot()
fig.show()

sys = bond_graph.to_control_ss()
x0_val = np.zeros_like((sys.nstates, 1))

time_response: ctrl.TimeResponseData = ctrl.step_response(sys, T=0.5, X0=x0_val)
T, yout, xout = time_response.time, time_response.outputs, time_response.states

yout = np.squeeze(yout) # yout is n_outputs x 1 x n_timesteps as the C matrix is has a shape of (n_outputs, n_states), so we need to squeeze the output to n_outputs x n_timesteps
xout = np.squeeze(xout) # xout is n_states x 1 x n_timesteps as the A matrix is has a shape of (n_states, n_states), so we need to squeeze the output to n_states x n_timesteps

print(f"Max time step: {np.max(np.diff(T))}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)

ax1: plt.Axes
ax2: plt.Axes
ax3: plt.Axes

ax1.set_xlabel("time (s)")
ax1.set_ylabel("Efforts (V, N, Nm, Pa)")
ax2.set_ylabel("Currents (A, m/s, rad/s, m^3/s)")
ax3.set_ylabel("States (Wb for p_elec and rad for p_mech)")

lines = []

for i, signal in enumerate(yout):
    if i < n_outputs // 2:
        l = ax1.plot(T, signal, label=f"e_{i}")
        lines.extend(l)
    else:
        l = ax2.plot(T, signal, label=f"f_{i - n_outputs // 2}")
        lines.extend(l)

l = ax3.plot(T, xout.T, label=[sp.pretty(st) for st in bond_graph.state_vars])
lines.extend(l)

#https://matplotlib.org/stable/gallery/event_handling/legend_picking.html
legend_lines = []
for ax in [ax1, ax2, ax3]:
    legend = ax.legend()
    legend_lines.extend(legend.get_lines())

map_legend_to_ax: dict[plt.Line2D, plt.Line2D] = {}
for legend_line, ax_line in zip(legend_lines, lines):
    legend_line.set_picker(5)
    map_legend_to_ax[legend_line] = ax_line

def on_pick(event):
    # On the pick event, find the original line corresponding to the legend
    # proxy line, and toggle its visibility.
    legend_line = event.artist

    # Do nothing if the source of the event is not a legend line.
    if legend_line not in map_legend_to_ax:
        return

    ax_line = map_legend_to_ax[legend_line]
    visible = not ax_line.get_visible()
    ax_line.set_visible(visible)
    
    # Rescale to visible data only
    ax_line.axes.relim(visible_only=True)
    ax_line.axes.autoscale_view(scalex=True, scaley=True)

    # Change the alpha on the line in the legend, so we can see what lines
    # have been toggled.
    legend_line.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()
