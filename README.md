# pyBondGraph
**pyBondGraph** is a Python library for **modeling and analyzing linear bond graph systems** using symbolic computation.

The library allows users to construct bond graph models programmatically, automatically derive the governing equations, and analyze the resulting dynamic systems using tools from control theory.

Bond graphs provide a **domain-independent modeling framework** for physical systems. Using a unified representation of power exchange, the same modeling approach can be used for electrical, mechanical, hydraulic, and multi-domain systems.

---

# Features

* Programmatic construction of **bond graph models**
* Automatic **symbolic equation derivation** using SymPy
* Conversion of models to **state-space systems**
* Example models for electrical and mechanical systems

---

# Installation

## Install directly from GitHub
The easiest way to install the library is directly via pip:
```bash
pip install git+https://github.com/MrP123/BondgraphSimulator.git
```
This installs the latest version of the package from the repository.

---

## Development installation
To work with the source code:
```bash
git clone https://github.com/MrP123/BondgraphSimulator.git
cd BondgraphSimulator
pip install -e .
```

---

# Dependencies

The main dependencies are:

* `sympy`
* `numpy`
* `networkx`
* `matplotlib`
* `control` only needed for the examples

Optional dependencies are used for experimental visualization tools.

---

# Basic Usage
A bond graph model is constructed by creating elements and connecting them via bonds.

Simple RC-Filter circuit:

```python
from pyBondGraph import BondGraph, SourceEffort, Resistor, Capacitor, OneJunction, Bond, Causality

bg = BondGraph()

# create elements
voltage_source = SourceEffort("U", "u_in")
resistor = Resistor("R", "R")
capacitor = Capacitor("C", "C")
series_junction = OneJunction("J1")

# connect elements
# causalities need to be assigned manually 
bg.add_bond(Bond(voltage_source, series_junction, Causality.EFFORT_OUT))
bg.add_bond(Bond(series_junction, resistor, Causality.EFFORT_OUT))
bg.add_bond(Bond(series_junction, capacitor, Causality.FLOW_OUT))

# plot the resulting BondGraph
bg.plot()

# derive system equations in linear state space form
A, B, C, D, x, n_states, n_inputs, n_outputs = bond_graph.get_state_space()
```

The library automatically derives the **symbolic system equations** describing the dynamics of the model.

---


# Core Concepts
Bond graphs represent **power exchange between system components**, where power is the product of **effort** and **flow** associated with the following components:

## Elements
| Element | Meaning                                  |
|---------|------------------------------------------|
| R       | Dissipation                              |
| C       | Energy storage (compliance, capacitance) |
| I       | Energy storage (inertia, inductance)     |
| Se      | Effort source                            |
| Sf      | Flow source                              |

## Junctions
| Junction | Meaning       |
|----------|---------------|
|     0    | Common effort |
|     1    | Common flow   |

## Sensors
| Sensor                 | Meaning                                     |
|------------------------|---------------------------------------------|
| IntegratedEffortSensor | Measures integral of the effort at its bond |
| IntegratedFlowSensor   | Measures integral of the flow at its bond   |

In mechanical bond graph models:
* **flow** corresponds to **velocity**
* **effort** corresponds to **force**

An **integrated flow sensor** can therefore be used to compute **position**:

---

# Example Systems
The repository contains example models illustrating typical applications of bond graphs.

### RLC Circuit
Demonstrates modeling of an electrical circuit using bond graph elements.

### DC Motor
A multi-domain electromechanical system coupling electrical and mechanical dynamics.

### Transformer
Example of energy transformation between two ports.

### Two DOF Mass–Spring–Damper System
Classical mass-spring-damper system with two degrees of freedom.

---

# Typical Applications
Bond graph modeling is particularly useful for:

* electromechanical systems
* robotics and mechatronics
* multi-domain energy systems
* control system modeling
* teaching system dynamics