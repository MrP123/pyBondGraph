from pyBondGraph import BondGraph, Bond, SourceEffort, SourceFlow, Capacitor, Resistor, OneJunction, ZeroJunction, Causality

import sympy as sp
import matplotlib.pyplot as plt
import control as ctrl
import numpy as np
import itertools

#%% Bondgraph modelling
# based on Fig. 1: https://www.researchgate.net/publication/383561812_Application_of_Simple_2R2C_Model_on_Large-Scale_Smart_Thermostat_Data

bond_graph = BondGraph()

To = SourceEffort("To", "To")

Re = Resistor("Re", "Re")
Ce = Capacitor("Ce", "Ce")
q_e = SourceFlow("q_e", "q_e")

Ri = Resistor("Ri", "Ri")
Ci = Capacitor("Ci", "Ci")
q_i = SourceFlow("q_i", "q_i")

j1_e = OneJunction("J1_e")
j0_e = ZeroJunction("J0_e")

j1_i = OneJunction("J1_i")
j0_i = ZeroJunction("J0_i")

# Ambient temperature as source effort to 1st R-C-branch
bond_graph.add_bond(Bond(To, j1_e, Causality.EFFORT_OUT))

# 1st R-C-branch with add. heat input
bond_graph.add_bond(Bond(j1_e, Re, Causality.EFFORT_OUT))
bond_graph.add_bond(Bond(j1_e, j0_e, Causality.FLOW_OUT))
bond_graph.add_bond(Bond(j0_e, Ce, Causality.FLOW_OUT))
bond_graph.add_bond(Bond(q_e, j0_e, Causality.FLOW_OUT))

# Connection between 1st & 2nd R-C-branch
bond_graph.add_bond(Bond(j0_e, j1_i, Causality.EFFORT_OUT))

# 2nd R-C-branch with add. heat input
bond_graph.add_bond(Bond(j1_i, Ri, Causality.EFFORT_OUT))
bond_graph.add_bond(Bond(j1_i, j0_i, Causality.FLOW_OUT))
bond_graph.add_bond(Bond(j0_i, Ci, Causality.FLOW_OUT))
bond_graph.add_bond(Bond(q_i, j0_i, Causality.FLOW_OUT))

# Connection to 3rd R-C-branch could follow here....

fig, _ = bond_graph.plot()
fig.show()

A, B, C, D, x, n_states, n_inputs, n_outputs = bond_graph.get_state_space()

# Print results
print("State vector x:")
sp.pprint(x)
print("Matrix A:")
sp.pprint(A)
print("\nMatrix B:")
sp.pprint(B)
print("\nMatrix C:")
sp.pprint(C)
print("\nMatrix D:")
sp.pprint(D)
print(f"Inputs: {bond_graph.inputs}")


#%% Numeric solving

# From the paper
Ri_val = 4.1e-4
Re_val = 1.4e-2
Ci_val = 9.6e6
Ce_val = 8.9e7

# ToDo: maybe make numeric value part of the element class?
subs_dict = {
    Re.value: Re_val,
    Ri.value: Ri_val,
    Ce.value: Ce_val,
    Ci.value: Ci_val
}

def to_numpy(M: sp.Matrix, subs: dict) -> np.ndarray:
    return np.array(M.subs(subs), dtype=np.float64)

# numpy materices with subsituted values for numeric simulation
A_mat_val = to_numpy(A, subs_dict)
B_mat_val = to_numpy(B, subs_dict)
C_mat_val = to_numpy(C, subs_dict)
D_mat_val = to_numpy(D, subs_dict)


lambda_real_abs = np.abs(np.real(np.linalg.eigvals(A_mat_val)))
stiffness_ratio = np.max(lambda_real_abs) / np.min(lambda_real_abs[np.nonzero(lambda_real_abs)])
print(f"Stiffness ratio: {stiffness_ratio:.2e}")

# simulate for 10 days with a time step of 1 hour
t_sim = np.linspace(0, 10*24*3600, int((10*24*3600) / (1*3600) + 1))
print(t_sim.shape)

# sinusoidal ambient temperature with a period of 24h, no solar heating and no internal heat generation
u_sim = np.zeros((n_inputs, t_sim.shape[0]))
u_sim[0, :] = 8 * np.sin(2*np.pi*t_sim/(24*3600)) + 16  # sinusoidal ambient temperature with a period of 24h
u_sim[1, :] = 0.0  # no solar heating
u_sim[2, :] = 0.0  # no internal heat generation

# inital state: 16°C in both capacitors
x0_val = 16.0 * np.asarray([Ce_val, Ci_val]) # [q] = J --> [C] = J/K --> T*C = q

sys = ctrl.ss(A_mat_val, B_mat_val, C_mat_val, D_mat_val)
time_response: ctrl.TimeResponseData = ctrl.forced_response(sys, T=t_sim, U=u_sim, X0=x0_val)

T, yout, xout = time_response.time, time_response.outputs, time_response.states
yout = np.squeeze(yout)  # yout is n_outputs x 1 x n_timesteps as the C matrix is has a shape of (n_outputs, n_states), so we need to squeeze the output to n_outputs x n_timesteps

print(f"Max time step: {np.max(np.diff(T))}")

#%% Plotting

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)

ax1: plt.Axes
ax2: plt.Axes
ax3: plt.Axes

ax1.set_xlabel("time (s)")
ax1.set_ylabel("Efforts (°C)")
ax2.set_ylabel("Flows (W)")
ax3.set_ylabel("States (J for q)")

def bond_num_to_indices(bond_num: int) -> tuple[int, int]:
    """Helper function to convert bond number to effort and flow indices in the output vector."""
    effort_index = bond_num
    flow_index = bond_num + n_outputs // 2
    return effort_index, flow_index

def plot_bond(bond_num: int):
    ei, fi = bond_num_to_indices(bond_num)
    ax1.plot(T, yout[ei], label=f"e_{bond_num}")
    ax2.plot(T, yout[fi], label=f"f_{bond_num}")


# Plot all efforts and flows with markers to distinguish them, but only every 25th point to avoid cluttering the plot
# shitty marker cycle 
# markers_effort, markers_flow = itertools.tee(itertools.cycle(("+", "o", "*", "x", "s", "d")), 2) #cycle is not resettable --> 2 iterators
# start_marker = itertools.cycle([11, 19, 23, 31, 41, 47, 53, 61, 71, 79, 83, 89, 97])

# for i, signal in enumerate(yout):
#     if i < n_outputs // 2:
#         ax1.plot(T, signal, label=f"e_{i}", marker=next(markers_effort), markevery=(next(start_marker), 25))
#     else:
#         ax2.plot(T, signal, label=f"f_{i - n_outputs // 2}", marker=next(markers_flow), markevery=(next(start_marker), 25))

# plot only bonds of interest
plot_bond(0) # To -> J1_e
plot_bond(3) # J0_e -> Ce
plot_bond(8) # J0_i -> Ci

ax3.plot(T, xout.T, label=[sp.pretty(st) for st in bond_graph.state_vars])

ax1.legend()
ax2.legend()
ax3.legend()
plt.show()