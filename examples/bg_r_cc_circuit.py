from pyBondGraph import BondGraph, Bond, SourceEffort, Capacitor, Resistor,  OneJunction, ZeroJunction

import sympy as sp
import matplotlib.pyplot as plt

bond_graph = BondGraph()

voltage_source = SourceEffort("U", "U0")
resistor = Resistor("R", "R")
capacitor1 = Capacitor("C1", "C1")
capacitor2 = Capacitor("C2", "C2")

junction1_SeR = OneJunction("J1")
junction0_CC = ZeroJunction("J0_CC")

bond_graph.add_bond(Bond(voltage_source, junction1_SeR, "effort_out"))
bond_graph.add_bond(Bond(junction1_SeR, resistor, "effort_out"))
bond_graph.add_bond(Bond(junction1_SeR, junction0_CC, "flow_out"))
bond_graph.add_bond(Bond(junction0_CC, capacitor1, "flow_out"))
bond_graph.add_bond(Bond(junction0_CC, capacitor2, "effort_out")) # If parallel then not causal anymore --> issue!

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
