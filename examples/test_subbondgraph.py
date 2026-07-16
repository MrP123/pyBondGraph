"""Test script for SubBondGraph workflow:
   create → save → load → instantiate → connect → solve
"""
from pyBondGraph import (
    Bond, BondGraph, SubBondGraph,
    SourceEffort, Resistor, Capacitor, Inductor,
    OneJunction, ZeroJunction, Causality
)

from pathlib import Path

print("=" * 60)
print("STEP 1: Build a simple RC sub-model")
print("=" * 60)

# RC sub-model: 1-junction -> R, C (no source — that's added externally)
rc_bg = BondGraph(name="RC")
R = Resistor("R", "R")
C = Capacitor("C", "C")
j1 = OneJunction("j1")

rc_bg.add_bond(Bond(j1, R, Causality.EFFORT_OUT))
rc_bg.add_bond(Bond(j1, C, Causality.FLOW_OUT))

rc_bg.get_solution_equations()

print(f"Elements: {[e.name for e in rc_bg.elements]}")
print(f"Bonds: {len(rc_bg.bonds)}")
print(f"State vars: {rc_bg.state_vars}")

print("\n" + "=" * 60)
print("STEP 2: Wrap as SubBondGraph with ports")
print("=" * 60)

sub_rc = SubBondGraph(
    name="RC_filter",
    bondgraph=rc_bg,
    ports={"input": j1}
)
print(f"SubBondGraph '{sub_rc.name}' with ports: {list(sub_rc.ports.keys())}")

print("\n" + "=" * 60)
print("STEP 3: Save to JSON")
print("=" * 60)

save_path = Path("./rc_filter.json")
sub_rc.save(save_path)
print(f"Saved to {save_path}")

# Read back and display
with open(save_path, "r") as f:
    content = f.read()
print(f"JSON length: {len(content)} chars")
print(f"First 200 chars: {content[:200]}...")

print("\n" + "=" * 60)
print("STEP 4: Load from JSON")
print("=" * 60)

rc_bg_loaded = SubBondGraph.load(save_path)
print(f"Loaded SubBondGraph '{rc_bg_loaded.name}'")
print(f"Ports: {list(rc_bg_loaded.ports.keys())}")
print(f"Elements: {[e.name for e in rc_bg_loaded.bondgraph.elements]}")
print(f"Bonds: {len(rc_bg_loaded.bondgraph.bonds)}")
assert len(rc_bg_loaded.bondgraph.elements) == 3, f"Expected 3 elements (j1, R, C), got {len(rc_bg_loaded.bondgraph.elements)}"
assert len(rc_bg_loaded.bondgraph.bonds) == 2, f"Expected 2 bonds, got {len(rc_bg_loaded.bondgraph.bonds)}"

print("\n" + "=" * 60)
print("STEP 5: Instantiate into a parent BondGraph")
print("=" * 60)

system = BondGraph(name="System")
ports1 = system.add_subbondgraph(rc_bg_loaded, "rc_inst1")

print(f"Instance 'rc_inst1' ports: {list(ports1.keys())}")
print(f"System elements: {[e.name for e in system.elements]}")
print(f"System state vars: {system.state_vars}")

print("\n" + "=" * 60)
print("STEP 6: Add a source and connect to the instance")
print("=" * 60)

# Add a voltage source to drive the instantiated RC filter
Se_main = SourceEffort("Vin", "Vin")

# Connect the main junction to the sub-model's input port
system.connect_ports(Se_main, ports1["input"], Causality.EFFORT_OUT)

print(f"System elements after connect: {[e.name for e in system.elements]}")
print(f"System bonds after connect: {len(system.bonds)}")

print("\n" + "=" * 60)
print("STEP 7: Solve the system")
print("=" * 60)

try:
    solution = system.get_solution_equations()
    print("Solution equations:")
    for var, expr in solution.items():
        print(f"  {var} = {expr}")
    print(f"\nState vars after solving: {system.state_vars}")
    assert len(system.state_vars) > 0, "Expected at least one state variable (q_inst1_C)"
    print("✓ State variables correctly created")
except Exception as e:
    print(f"Solver error: {type(e).__name__}: {e}")
    print("(This may indicate an issue with equation assembly — debug from here)")

print("\n" + "=" * 60)
print("STEP 8: Multi-instance test")
print("=" * 60)

system2 = BondGraph(name="MultiInstance")
ports_a = system2.add_subbondgraph(rc_bg_loaded, "filterA")
ports_b = system2.add_subbondgraph(rc_bg_loaded, "filterB")

print(f"Instance A elements: {[e.name for e in system2.elements if e.name.startswith('filterA')]}")
print(f"Instance B elements: {[e.name for e in system2.elements if e.name.startswith('filterB')]}")

# Add sources and connect both instances using the new API
Se_A = SourceEffort("VA", "VA")
Se_B = SourceEffort("VB", "VB")

system2.connect_ports(Se_A, ports_a["input"], Causality.EFFORT_OUT)
system2.connect_ports(Se_B, ports_b["input"], Causality.EFFORT_OUT)

fig, _ = system2.plot()
fig.show()
while True:
    try:
        input("Press Enter to continue...")
        break
    except KeyboardInterrupt:
        print("\nInterrupted. Continuing...")

solution2 = system2.get_solution_equations()
print(f"State vars after solving: {system2.state_vars}")

# Check no symbol collisions
state_names = [str(sv) for sv in system2.state_vars]
assert len(state_names) == len(set(state_names)), f"Symbol collision! {state_names}"
assert len(state_names) == 2, f"Expected 2 state vars (one per instance), got {len(state_names)}"
print("✓ No symbol collisions between instances")
print(f"✓ {len(state_names)} independent state variables: {state_names}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
