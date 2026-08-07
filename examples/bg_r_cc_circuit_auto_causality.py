from pyBondGraph import BondGraph, SourceEffort, Capacitor, Resistor,  OneJunction, ZeroJunction

import sympy as sp
import matplotlib.pyplot as plt

bond_graph = BondGraph()

voltage_source = SourceEffort("U", "U0")
resistor = Resistor("R", "R")
capacitor1 = Capacitor("C1", "C1")
capacitor2 = Capacitor("C2", "C2")

junction1_SeR = OneJunction("J1")
junction0_CC = ZeroJunction("J0_CC")

bond_graph.connect(voltage_source, junction1_SeR)
bond_graph.connect(junction1_SeR, resistor)
bond_graph.connect(junction1_SeR, junction0_CC)
bond_graph.connect(junction0_CC, capacitor1)
bond_graph.connect(junction0_CC, capacitor2) # If parallel then not causal anymore --> issue!

bond_graph.assign_causality()

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
