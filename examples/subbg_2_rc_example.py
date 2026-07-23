from pyBondGraph import (
    BondGraph,
    SubBondGraph,
    SourceEffort,
    ZeroJunction,
    Causality,
)

from pathlib import Path

# Create a bond graph with two instances of the same RC filter sub-bondgraph connected
# This is NOT a 2nd order RC filter, as that would require a structure like Se --> 1 --> R_A, 0 --> C_A, 1 --> R_B, 0 --> C_B
system = BondGraph(name="2 RC example")
rc_system = SubBondGraph.load(Path(__file__).with_name("rc_filter.json"))
ports_a = system.add_subbondgraph(rc_system, "A", is_prefix=False)
ports_b = system.add_subbondgraph(rc_system, "B", is_prefix=False)
zero_junction = ZeroJunction("j0")

# Add source and connect both instances using the new API
Se_A = SourceEffort("V_in", "V_in")

system.connect(Se_A, ports_a["input"], Causality.EFFORT_OUT)
system.connect(ports_a["input"], zero_junction, Causality.FLOW_OUT)
system.connect(zero_junction, ports_b["input"], Causality.EFFORT_OUT)

fig, _ = system.plot()
fig.show()

#%% Get state space representation and print it

A, B, C, D, x, n_states, n_inputs, n_outputs = system.get_state_space()

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

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
print(f"Inputs: {system.inputs}")


#%% Numeric solving

# ToDo: maybe make numeric value part of the element class?
subs_dict = {
    "R_A": 10e3,
    "R_B": 10e3,
    "C_A": 100e-9,
    "C_B": 100e-9,
}

def to_numpy(M: sp.Matrix, subs: dict) -> np.ndarray:
    subs = {sp.symbols(key, real=True, positive=True): value for key, value in subs.items() if isinstance(key, str)}
    
    return np.array(M.subs(subs), dtype=np.float64)

# numpy materices with subsituted values for numeric simulation
A_mat_val = to_numpy(A, subs_dict)
B_mat_val = to_numpy(B, subs_dict)
C_mat_val = to_numpy(C, subs_dict)
D_mat_val = to_numpy(D, subs_dict)


lambda_real_abs = np.abs(np.real(np.linalg.eigvals(A_mat_val)))
stiffness_ratio = np.max(lambda_real_abs) / np.min(lambda_real_abs[np.nonzero(lambda_real_abs)])
print(f"Stiffness ratio: {stiffness_ratio:.2e}")

dt = 1e-6
t_sim = np.arange(0, 20e-3 + dt, dt)
print(t_sim.shape)

u_sim = np.zeros((n_inputs, t_sim.shape[0]))
u_sim[0, :] = 1  # immediate step input

#intial state vector
x0_val = np.zeros(n_states)

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
ax1.set_ylabel("Efforts (V)")
ax2.set_ylabel("Flows (A)")
ax3.set_ylabel("States (C)")

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
import itertools
markers_effort, markers_flow = itertools.tee(itertools.cycle(("+", "o", "*", "x", "s", "d")), 2) #cycle is not resettable --> 2 iterators
start_marker = itertools.cycle([11, 19, 23, 31, 41, 47, 53, 61, 71, 79, 83, 89, 97])

for i, signal in enumerate(yout):
    if i < n_outputs // 2:
        ax1.plot(T, signal, label=f"e_{i}", marker=next(markers_effort), markevery=(next(start_marker), 25))
    else:
        ax2.plot(T, signal, label=f"f_{i - n_outputs // 2}", marker=next(markers_flow), markevery=(next(start_marker), 25))

# plot only bonds of interest
#plot_bond(0) # To -> J1_e
#plot_bond(3) # J0_e -> Ce
#plot_bond(8) # J0_i -> Ci

ax3.plot(T, xout.T, label=[sp.pretty(st) for st in system.state_vars])

ax1.legend()
ax2.legend()
ax3.legend()
plt.show()