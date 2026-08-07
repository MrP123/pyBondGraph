# pyBondGraph
**pyBondGraph** is a Python library for **modeling and analyzing linear bond graph systems** using symbolic computation.

The library allows users to construct bond graph models programmatically, automatically derive the governing equations, and analyze the resulting dynamic systems using tools from control theory.

Bond graphs provide a **domain-independent modeling framework** for physical systems. Using a unified representation of power exchange, the same modeling approach can be used for electrical, mechanical, hydraulic, and multi-domain systems.

---

# Features

* Programmatic construction of **bond graph models**
* **Automatic causality assignment** via SCAP (Sequential Causality Assignment Procedure), with optional manual override or mixed mode
* Automatic **symbolic equation derivation** using SymPy
* Conversion of models to **linear state-space systems** ($\dot{x} = Ax + Bu$, $y = Cx + Du$)
* **Composable sub-models** via `SubBondGraph` with deep-copy namespace isolation
* **Two-port elements**: Transformer and Gyrator with automatic causality propagation
* **Sensor elements**: `IntegratedEffortSensor` and `IntegratedFlowSensor` for measuring integrated generalized variables (e.g. position from velocity)
* **Domain-neutral aliases**: `Compliance` = `Capacitor`, `Inertance` = `Inductor`, `Resistance` = `Resistor`
* Example models for electrical and electromechanical systems
* Integration with **python-control** for numerical simulation (step response, Bode plots, etc.)

---

# Installation

## Install from PyPI or Github
The easiest way to install the library is via your preferred package manager (e.g. pip) directly from PyPI:
```bash
pip install pyBondGraph
```
Alternatively one can install the latest development version directly from the GitHub repository:
```bash
pip install git+https://github.com/MrP123/pyBondGraph.git
```

---

## Development installation
To work with the source code:
```bash
git clone https://github.com/MrP123/pyBondGraph.git
cd pyBondGraph
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
A bond graph model is constructed by creating elements and connecting them via the `connect()` convenience method, which creates bonds and adds them to the graph in one step.

## RC-Filter with automatic causality (SCAP)

```python
from pyBondGraph import BondGraph, SourceEffort, Resistor, Capacitor, OneJunction

bg = BondGraph()

# create elements
voltage_source = SourceEffort("U", "u_in")
resistor = Resistor("R", "R")
capacitor = Capacitor("C", "C")
series_junction = OneJunction("J1")

# connect elements --> causality is assigned automatically by SCAP
bg.connect(voltage_source, series_junction)
bg.connect(series_junction, resistor)
bg.connect(series_junction, capacitor)

# plot the resulting BondGraph
bg.plot()

# derive system equations in linear state space form
A, B, C, D, x, n_states, n_inputs, n_outputs = bg.get_state_space()
```

Causality can also be assigned **manually** by passing a `Causality` value to `connect()`, or in **mixed mode** where some bonds are fixed and SCAP resolves the rest.

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

## Two-Port Elements
| Element     | Meaning                                                    |
|-------------|------------------------------------------------------------|
| TF          | Transformer — same causality on both bonds                 |
| GY          | Gyrator — opposite causality on both bonds                 |

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
* **flow** corresponds to **velocity**, i.e. an *IntegratedFlowSensor* can be used to compute **position**.
* **effort** corresponds to **force**

---

# Example Systems
The repository contains example models illustrating typical applications of bond graphs.

### RLC Circuit
Demonstrates modeling of an electrical circuit using bond graph elements.

### DC Motor
A multi-domain electromechanical system coupling electrical and mechanical dynamics. Also demonstrates integration with the `python-control` package for numerical simulation (step response).

### Transformer
Example of energy transformation between two ports.

### Two DOF Mass–Spring–Damper System
Classical mass-spring-damper system with two degrees of freedom.

---

# Causality Assignment

pyBondGraph supports three modes for assigning causality:

1. **Automatic (SCAP)** — omit causality in `connect()` calls; `assign_causality()` is called automatically when solving. The Sequential Causality Assignment Procedure assigns causality in priority order: sources, storage elements (integral causality), resistors, then propagation through junctions and two-port elements.
2. **Manual** — pass `Causality.EFFORT_OUT` or `Causality.FLOW_OUT` explicitly to each `connect()` call.
3. **Mixed** — fix causality on some bonds, let SCAP resolve the rest.

If a storage element cannot receive integral causality (which would imply a DAE rather than an ODE), a `DerivativeCausalityError` is raised with a clear diagnostic message.

---

# Typical Applications
Bond graph modeling is particularly useful for:

* electromechanical systems
* robotics and mechatronics
* multi-domain energy systems
* control system modeling
* teaching system dynamics

---

# Planned Features

* **FMU Export**: export bond graph models as Functional Mock-up Units (FMI standard) for interoperability with Simulink, Dymola, OpenModelica, and other FMI-compliant tools
* **Nonlinear element support**: general nonlinear constitutive laws with Jacobian linearization
* **Convenience bridge to python-control**: `to_control_ss(params)` method wrapping the existing manual pattern
