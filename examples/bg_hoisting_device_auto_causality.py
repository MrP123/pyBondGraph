from pyBondGraph import BondGraph, Causality, SourceEffort, Inductor, Capacitor, Resistor, OneJunction, ZeroJunction, Gyrator, Transformer, IntegratedFlowSensor
from pyBondGraph.core import Junction

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

import itertools

bond_graph = BondGraph()

voltage_source = SourceEffort("V", "U_src")
junction_elec = OneJunction("J1_1")
inductor = Inductor("I_elec", "L")
resistor = Resistor("R_elec", "R_el")
gyrator = Gyrator("G1", "K")
junction_mech = OneJunction("J1_2")
bearing = Resistor("R_mech", "R_b")
inertia = Inductor("I_mech", "J")
transformer = Transformer("W", "r")
junction_rope = ZeroJunction("J0_1")
elastic_rope = Capacitor("C_r", "C_r")
junction_mass = OneJunction("J1_3")
gravity = SourceEffort("g", "G")
mass = Inductor("I_mass", "m")

displacement_sensor = IntegratedFlowSensor("x_mass")

bond_graph.connect(voltage_source, junction_elec)
bond_graph.connect(junction_elec, resistor)
bond_graph.connect(junction_elec, inductor)
bond_graph.connect(junction_elec, gyrator)
bond_graph.connect(gyrator, junction_mech)
bond_graph.connect(junction_mech, bearing)
bond_graph.connect(junction_mech, inertia)
bond_graph.connect(junction_mech, transformer)
bond_graph.connect(transformer, junction_rope)
bond_graph.connect(junction_rope, elastic_rope)
bond_graph.connect(junction_rope, junction_mass)
bond_graph.connect(gravity, junction_mass)
bond_graph.connect(junction_mass, mass)

bond_graph.connect(junction_mass, displacement_sensor)

A, B, C, D, x, n_states, n_inputs, n_outputs = bond_graph.get_state_space()

#Print results
print("x_vec:")
sp.pprint(x)
print("u_vec:")
sp.pprint(bond_graph.inputs)
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


U_src_val = 5.0
R_el_val = 4
L_val = 15e-6
K_t_val = 9.54e-3
J_val = 1e-6
R_b_val = 1e-6
d_winch = 100e-3
c_rope = 1e-6
m = 5e-3  # kg

# ToDo: maybe make numeric value part of the element class?
subs_dict = {
    voltage_source.value: U_src_val,
    inductor.value: L_val,
    resistor.value: R_el_val,
    gyrator.value: K_t_val,
    bearing.value: R_b_val,
    inertia.value: J_val,
    transformer.value: d_winch / 2,
    elastic_rope.value: c_rope,
    gravity.value: m * 9.81,
    mass.value: m
}

# assert len(subs_dict) == len([elem for elem in bond_graph.elements if not isinstance(elem, Junction)]), "All elements (except junctions) must have numeric values for simulation!"

def to_numpy(M: sp.Matrix, subs: dict) -> np.ndarray:
    return np.array(M.subs(subs), dtype=np.float64)

A_mat_val = to_numpy(A, subs_dict)
B_mat_val = to_numpy(B, subs_dict)
C_mat_val = to_numpy(C, subs_dict)
D_mat_val = to_numpy(D, subs_dict)

lambda_real_abs = np.abs(np.real(np.linalg.eigvals(A_mat_val)))
stiffness_ratio = np.max(lambda_real_abs) / np.min(lambda_real_abs[np.nonzero(lambda_real_abs)])
print(f"Stiffness ratio: {stiffness_ratio:.2e}")

x0_val = np.zeros_like(A_mat_val[0, :])

t_sim = np.linspace(0, 5, 10000)
u_sim = np.zeros((n_inputs, t_sim.shape[0]))

# constant value inputs --> step
u_sim[0, 0:5000] = 0.0
u_sim[0, 5000:] = U_src_val
u_sim[1, :] = -m * 9.81  # gravity input

sys = ctrl.ss(A_mat_val, B_mat_val, C_mat_val, D_mat_val)
time_response: ctrl.TimeResponseData = ctrl.forced_response(sys, T=t_sim, U=u_sim, X0=x0_val)

T, yout, xout = time_response.time, time_response.outputs, time_response.states
yout = np.squeeze(yout)  # yout is n_outputs x 1 x n_timesteps as the C matrix is has a shape of (n_outputs, n_states), so we need to squeeze the output to n_outputs x n_timesteps

print(f"Max time step: {np.max(np.diff(T))}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)

ax1: plt.Axes
ax2: plt.Axes
ax3: plt.Axes

ax1.set_xlabel("time (s)")
ax1.set_ylabel("Efforts (V, N, Nm, Pa)")
ax2.set_ylabel("Currents (A, m/s, rad/s, m^3/s)")
ax3.set_ylabel("States")

markers_effort, markers_flow = itertools.tee(itertools.cycle(("+", "o", "*", "x", "s", "d")), 2) #cycle is not resettable --> 2 iterators
start_marker = itertools.cycle([11, 19, 23, 31, 41, 47, 53, 61, 71, 79, 83, 89, 97])

for i, signal in enumerate(yout):
    if i < n_outputs // 2:
        ax1.plot(T, signal, label=f"e_{i}", marker=next(markers_effort), markevery=(next(start_marker), 500))
    else:
        ax2.plot(T, signal, label=f"f_{i - n_outputs // 2}", marker=next(markers_flow), markevery=(next(start_marker), 500))

ax3.plot(T, xout.T, label=[sp.pretty(st) for st in bond_graph.state_vars])

f_sensor = yout[-1]
q_sensor = np.cumsum(f_sensor) * (T[1] - T[0])  # integral of flow = displacement

ax3.plot(T, q_sensor, label="x_mass manually integrated", linestyle="--")

ax1.legend()
ax2.legend()
ax3.legend()
plt.show()