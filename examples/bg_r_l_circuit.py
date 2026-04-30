from pyBondGraph import BondGraph, Bond, SourceEffort, Inductor, Resistor,  OneJunction, ZeroJunction

import sympy as sp
import matplotlib.pyplot as plt

bond_graph = BondGraph()

voltage_source = SourceEffort("U", "U0")
resistor1 = Resistor("R1", "R1")
resistor2 = Resistor("R2", "R2")
resistor3 = Resistor("R3", "R3")
inductor = Inductor("I", "L")

junction0_RL = ZeroJunction("J0_RL")
junction0_RR = ZeroJunction("J0_RR")
junction1 = OneJunction("J1")

bond_graph.add_bond(Bond(voltage_source, junction0_RL, "effort_out"))
bond_graph.add_bond(Bond(junction0_RL, inductor, "effort_out"))
bond_graph.add_bond(Bond(junction0_RL, junction1, "effort_out"))
bond_graph.add_bond(Bond(junction1, resistor1, "flow_out"))
bond_graph.add_bond(Bond(junction1, junction0_RR, "effort_out"))
bond_graph.add_bond(Bond(junction0_RR, resistor2, "effort_out"))
bond_graph.add_bond(Bond(junction0_RR, resistor3, "effort_out"))

bond_graph.plot()
plt.show()

A, B, C, D, x, n_states, n_inputs, n_outputs = bond_graph.get_state_space()

# Print results
print("Matrix A:")
sp.pprint(A)
print("\nMatrix B:")
sp.pprint(B)
print("\nMatrix C:")
sp.pprint(C)
print("\nMatrix D:")
sp.pprint(D)