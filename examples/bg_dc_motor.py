from pyBondGraph import BondGraph, Causality, SourceEffort, Inductor, Resistor, OneJunction, Gyrator

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

bond_graph = BondGraph()

voltage_source = SourceEffort("V", "U_A(t)")
junction_elec = OneJunction("J1_1")
inductor = Inductor("I_elec", "L_A")
resistor = Resistor("R_elec", "R_A")
gyrator = Gyrator("G1", "K_t")
junction_mech = OneJunction("J1_2")
bearing = Resistor("R_mech", "R_B")
inertia = Inductor("I_mech", "J")

bond_graph.connect(voltage_source, junction_elec, Causality.EFFORT_OUT)
bond_graph.connect(junction_elec, resistor, Causality.FLOW_OUT)
bond_graph.connect(junction_elec, inductor, Causality.EFFORT_OUT)
bond_graph.connect(junction_elec, gyrator, Causality.FLOW_OUT)
bond_graph.connect(gyrator, junction_mech, Causality.EFFORT_OUT)
bond_graph.connect(junction_mech, bearing, Causality.FLOW_OUT)
bond_graph.connect(junction_mech, inertia, Causality.EFFORT_OUT)

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

resistor.numeric_value = 4
inductor.numeric_value = 15e-6
gyrator.numeric_value = 9.54e-3
inertia.numeric_value = 1e-6
bearing.numeric_value = 1e-6

A_mat_val, B_mat_val, C_mat_val, D_mat_val = bond_graph.get_numeric_state_space()

sys = bond_graph.to_control_ss()
print(sys.nstates)
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


for i, signal in enumerate(yout):
    if i < n_outputs // 2:
        ax1.plot(T, signal, label=f"e_{i}")
    else:
        ax2.plot(T, signal, label=f"f_{i - n_outputs // 2}")

ax3.plot(T, xout.T, label=[sp.pretty(st) for st in bond_graph.state_vars])

ax1.legend()
ax2.legend()
ax3.legend()
plt.show()
