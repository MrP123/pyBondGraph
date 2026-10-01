from pathlib import Path

from pyBondGraph import BondGraph, SourceEffort, Inductor, Resistor, OneJunction, Gyrator

bond_graph = BondGraph()

voltage_source = SourceEffort("V", "U_in")
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


bond_graph.to_fmu(model_name="DC_Motor", dest=Path(__file__).parent)
print("FMU generated successfully: DC_Motor.fmu")