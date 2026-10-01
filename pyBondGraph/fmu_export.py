"""Build-time generator for a self-contained Model-Exchange FMU of a bond graph.

The problem with keeping the symbolic matrices + ``sympy.lambdify`` inside the
slave's ``__init__`` is that *all* of that code runs at FMU **instantiation**
time -- i.e. inside Simulink. That forces the Python interpreter Simulink picks
up to have ``sympy`` AND ``pyBondGraph`` (and their dependencies) importable,
which is fragile and hard to reproduce outside the development ``.venv``.

This module instead does *all* symbolic work here, at build time, and emits a
self-contained slave module (``fmu_slave.py``) whose only runtime imports are
the standard library (``math``) and ``pythonfmu3`` (which the FMU bundles its
own runtime for). No sympy, no pyBondGraph at runtime.

Parameters stay genuinely live: the generated matrix functions take the
parameters as arguments, and the slave re-evaluates them every derivative step
from the current parameter attributes. So changing a parameter from Simulink
still changes the dynamics -- nothing is baked in at export time.

Usage::

    python pyBondGraph/fmu_export.py        # writes fmu_slave.py and builds the FMU
"""

import hashlib
import uuid
from pathlib import Path
from string import Template

import sympy as sp

_TEMPLATE_PATH = Path(__file__).resolve().parent / "fmu_export_template.py.tmpl"

from pyBondGraph import BondGraph, SourceEffort, Inductor, Resistor, OneJunction, Gyrator


# -----------------------------------------------------------------------------
# Model definition (the only hardcoded part)
# -----------------------------------------------------------------------------
def build_bond_graph() -> BondGraph:
    """Build and return the bond graph to be exported.

    Swap the body of this function to export a different model; everything
    downstream is fully generic.
    """
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

    return bond_graph


# -----------------------------------------------------------------------------
# Code generation
# -----------------------------------------------------------------------------
def _matrix_literal(M: sp.Matrix) -> str:
    """Render a sympy Matrix as a nested-list Python literal using pure math.

    ``sp.pycode`` emits plain arithmetic operators and ``math.``-qualified
    function calls, so the result evaluates with only the standard library.
    """
    if M.rows == 0 or M.cols == 0:
        return "[]"
    rows = []
    for i in range(M.rows):
        elems = [sp.pycode(M[i, j]) for j in range(M.cols)]
        rows.append("[" + ", ".join(elems) + "]")
    return "[" + ", ".join(rows) + "]"


def _matrix_fn_src(name: str, M: sp.Matrix, params) -> str:
    args = ", ".join(p.name for p in params)
    return f"def {name}({args}):\n    return {_matrix_literal(M)}\n"



def generate_slave_source(bond_graph: BondGraph, class_name: str = "BondGraphSlave", author_name: str = "MtP", description: str = "A bond graph model.") -> str:
    """Do all the symbolic work and return the source of a self-contained slave."""
    bond_graph.assign_causality()

    A, B, C, D, _, n_states, n_inputs, n_outputs = bond_graph.get_state_space()
    n_bonds = len(bond_graph.bonds)

    # Free parameters = every symbol in A/B/C/D that is not an input.
    free_vars = set()
    for Mtx in (A, B, C, D):
        free_vars |= Mtx.free_symbols
    for inp in bond_graph.inputs:
        free_vars.discard(inp)
    params = sorted(free_vars, key=lambda s: s.name)

    subs = bond_graph.get_substitution_dict()

    state_names = [sv.name for sv in bond_graph.state_vars]
    input_names = [iv.name for iv in bond_graph.inputs]
    param_names = [p.name for p in params]
    param_starts = {p.name: float(subs.get(p, 0.0)) for p in params}

    matrix_fns = "\n".join(
        _matrix_fn_src(fn, Mtx, params)
        for fn, Mtx in (("_A_fn", A), ("_B_fn", B), ("_C_fn", C), ("_D_fn", D))
    )

    # This is the GUID passed to the FMU builder, describing the model content.
    # Rebuilds of the same model will have the same GUID, but any change to the model will change it.
    fingerprint = "\0".join(
        [
            class_name,
            repr(state_names),
            repr(input_names),
            repr(param_names),
            repr(param_starts),
            matrix_fns,
        ]
    )
    guid = str(uuid.uuid5(uuid.NAMESPACE_OID, fingerprint))

    # ---- assemble the runtime module from the template file -----------------
    # This uses the Python standard library's string.Template, which is a simple $-substitution engine.
    # This needs repr to get proper Python literals for lists, etc.
    template = Template(_TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        state_names=repr(state_names),
        input_names=repr(input_names),
        param_names=repr(param_names),
        param_starts=repr(param_starts),
        n_states=n_states,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_bonds=n_bonds,
        matrix_fns=matrix_fns,
        class_name=class_name,
        author_name=author_name,
        description=description,
        guid=guid,
    )


# -----------------------------------------------------------------------------
# Build entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    here = Path(__file__).resolve().parent

    # Name generated module after the class --> FMU imports the class from this module
    # therefore, it must be available at runtime.
    # Simulnik uses one CPython interpreter for the whole session --> so if the module name is generic,
    # it will be cached --> this gives a runtime error (module ... has no attribute ...) if the class name changes but the module name is the same.
    class_name = "DC_Motor_BG"
    source = generate_slave_source(build_bond_graph(), class_name=class_name, description="Bond graph of a simple DC motor")

    # A similar thing is done for the whole FMU:
    # This hash encompases the fact whether the model OR the template used for generating it has changed.
    # This once again prevents the cache of returning a stale module
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    slave_path = here / f"fmu_{class_name}_{digest}.py"

    slave_path.write_text(source, encoding="utf-8")
    print(f"Generated self-contained slave: {slave_path}")

    # Build the FMU from the generated (dependency-free) slave module.
    from pythonfmu3 import FmuBuilder

    FmuBuilder.build_FMU(str(slave_path), dest=str(here.parent))
    print("FMU built.")

    slave_path.unlink()  # clean up the generated slave module
