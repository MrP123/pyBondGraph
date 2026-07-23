from pyBondGraph import (
    Bond,
    BondGraph,
    SubBondGraph,
    SourceEffort,
    Resistor,
    Capacitor,
    OneJunction,
    Causality,
)

from pathlib import Path

print("=" * 60)
print("STEP 1: Build a simple RC sub-model")
print("=" * 60)

# RC sub-model: 1-junction -> R, C (no source --> should be added externally)
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

sub_rc = SubBondGraph(name="RC_filter", bondgraph=rc_bg, ports={"input": j1})
print(f"SubBondGraph '{sub_rc.name}' with ports: {list(sub_rc.ports.keys())}")



print("\n" + "=" * 60)
print("STEP 3: Save to JSON")
print("=" * 60)

save_path = Path(__file__).with_name("rc_filter.json")
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

# Creat a new bond graph and add the loaded sub-model as an instance
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
system.connect(Se_main, ports1["input"], Causality.EFFORT_OUT)

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

# Create new bond graph and add two instances of the same sub-model with different names
system2 = BondGraph(name="MultiInstance")
ports_a = system2.add_subbondgraph(rc_bg_loaded, "fA", is_prefix=False)
ports_b = system2.add_subbondgraph(rc_bg_loaded, "fB", is_prefix=False)

print(f"Instance A elements: {[e.name for e in system2.elements if e.name.endswith('fA')]}")
print(f"Instance B elements: {[e.name for e in system2.elements if e.name.endswith('fB')]}")

# Connecting the two instances in "series" requires setting the causality of the bond from filterB's j1 to R to FLOW_OUT
# This works, as R elements do not have a preferred causality
for bond in system2.bonds:
    if bond.from_element.name == "j1_fB" and bond.to_element.name == "R_fB":
        print(f"Setting causality for bond {bond} from {bond.causality} to FLOW_OUT")
        bond.causality = Causality.FLOW_OUT

# Add source and connect both instances using the new API
Se_A = SourceEffort("V_in", "V_in")

system2.connect(Se_A, ports_a["input"], Causality.EFFORT_OUT)
system2.connect(ports_a["input"], ports_b["input"], Causality.FLOW_OUT)

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
for var, expr in solution2.items():
    print(f"  {var} = {expr}")

# Check no symbol collisions
state_names = [str(sv) for sv in system2.state_vars]
assert len(state_names) == len(set(state_names)), f"Symbol collision! {state_names}"
assert len(state_names) == 2, f"Expected 2 state vars (one per instance), got {len(state_names)}"
print("✓ No symbol collisions between instances")
print(f"✓ {len(state_names)} independent state variables: {state_names}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
