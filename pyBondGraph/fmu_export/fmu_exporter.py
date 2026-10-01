"""
Build-time generator for a self-contained Model-Exchange FMU of a bond graph.
To use, call BondGraph.to_fmu() or the standalone to_fmu() function.
"""
from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from string import Template

import sympy as sp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..bondgraph import BondGraph

_TEMPLATE_PATH = Path(__file__).resolve().parent / "fmu_export_template.py.tmpl"


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


def generate_slave_source(
    bond_graph: BondGraph,
    class_name: str = "BondGraphSlave",
    author_name: str = "MtP",
    description: str = "A bond graph model.",
) -> str:
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
# Public API
# -----------------------------------------------------------------------------
def to_fmu(
    bond_graph: BondGraph,
    dest: str | Path = ".",
    class_name: str = "BondGraphSlave",
    author_name: str = "MtP",
    description: str = "A bond graph model.",
    keep_slave_python_code: bool = False,
) -> Path:
    """Export a bond graph as a self-contained Model-Exchange FMU.

    All symbolic work (causality, state-space, free-parameter discovery) is done
    here at build time and emitted into a dependency-free slave module, which is
    then compiled into an FMI 3.0 FMU.
    The resulting FMU imports only ``math`` and ``pythonfmu3`` at runtime to be as 
    portable as possible for us in e.g. Simulink. Model parameters remain tunable at runtime.

    Parameters
    ----------
    bond_graph : BondGraph
        The bond graph to export. Must be solvable.
    dest : str or Path, optional
        Directory in which the ``.fmu`` file is written. Defaults to the current
        directory.
    class_name : str, optional
        Name of the generated slave class and of the FMU.
    author_name : str, optional
        Author recorded in the FMU metadata.
    description : str, optional
        Human-readable model description recorded in the FMU metadata.
    keep_slave_python_code : bool, optional
        If ``True`` the intermediate generated slave module is left on disk next
        to this file instead of being removed after the build (useful for
        debugging the generated code).

    Returns
    -------
    Path
        Path to the directory the FMU was written to.

    Raises
    ------
    ImportError
        If the ``pythonfmu3`` package is not installed.
    """
    try:
        from pythonfmu3 import FmuBuilder
    except ImportError:
        raise ImportError(
            "pythonfmu3 is required for to_fmu(). "
            "Install it with: pip install pythonfmu3"
        )

    here = Path(__file__).resolve().parent
    dest = Path(dest)

    source = generate_slave_source(
        bond_graph,
        class_name=class_name,
        author_name=author_name,
        description=description,
    )

    # The slave module name carries a short content hash --> changes whenever the
    # model OR the template changes, which keeps Simulink's cached CPython from
    # returning a stale module after a rebuild.
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    slave_path = here / f"fmu_{class_name}_{digest}.py"
    slave_path.write_text(source, encoding="utf-8")

    try:
        FmuBuilder.build_FMU(str(slave_path), dest=str(dest))
    finally:
        if not keep_slave_python_code:
            slave_path.unlink(missing_ok=True)

    return dest
