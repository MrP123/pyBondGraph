"""SubBondGraph — a reusable, connectable sub-model.

A SubBondGraph wraps a BondGraph together with named ports.
It can be instantiated (deep-copied with namespace prefixing) into
a parent BondGraph, and two sub-models can be connected at their
ports via a new bond.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from .bondgraph import BondGraph

from .core import (
    Bond,
    Causality,
    Node,
    Junction,
    ElementOnePort,
    ElementTwoPort,
    StatefulElement,
)
from .core import Port


class SubBondGraph:
    """A reusable bond-graph sub-model with named connection ports.

    Parameters
    ----------
    name : str
        Unique name for this sub-model.  Used as namespace prefix
        when the sub-model is instantiated into a parent graph
        (e.g. ``"Motor"`` produces symbols like ``e_Motor_0``).
    bondgraph : BondGraph
        The internal bond graph that defines this sub-model's
        physics.
    ports : dict[str, Node]
        Named mapping of port names to boundary :class:`Node` objects
        (e.g. ``{"in": j1_mech, "out": gyrator}``).  The type alias
        :data:`Port` is provided for convenience.
    """

    def __init__(self, name: str, bondgraph: BondGraph, ports: Port):
        self.name = name
        self.bondgraph = bondgraph
        self.ports = ports

        # Validate that every port node is actually in the bondgraph
        bg_elements = set(bondgraph.elements)
        for port_name, node in ports.items():
            if node not in bg_elements:
                raise ValueError(
                    f"Port '{port_name}' references node "
                    f"{node} which is not in the bond graph."
                )

    # ------------------------------------------------------------------
    # Instantiation (deep-copy with namespace prefixing)
    # ------------------------------------------------------------------

    def _instantiate(self, parent_bondgraph: BondGraph, instance_name: str | None = None, is_prefix: bool = True) -> Port:
        """Create a namespace-prefixed copy and merge it into *parent_bondgraph*.

        .. note::
            This is an internal method.  Prefer
            :meth:`BondGraph.add_subbondgraph` which delegates here
            and provides a cleaner, graph-centric API::

                ports = system.add_subbondgraph(sub, "inst1")

        Every element and bond is deep-copied.  Element names and bond
        symbols are prefixed with ``instance_name + "_"`` so that
        multiple instances of the same sub-model can coexist without
        symbol collisions.

        Parameters
        ----------
        parent_bondgraph : BondGraph
            The parent graph to merge into.
        instance_name : str | None, optional
            Override the namespace prefix.  Defaults to ``self.name``.

        Returns
        -------
        Port
            A mapping of port names to *new* Node objects whose
            elements belong to the parent graph.  Use these for
            subsequent :meth:`BondGraph.connect` calls.
        """

        if instance_name is None:
            instance_name = self.name

        def get_new_name(orig_name: str) -> str:
            """Return a new name for an element or symbol, with prefix/suffix."""
            if is_prefix:
                return instance_name + "_" + orig_name
            else:
                return orig_name + "_" + instance_name

        # 1. Deep-copy all elements and build old→new mapping
        elem_map: dict[int, Node] = {}  # id(old_element) → new_element
        for elem in self.bondgraph.elements:
            new_elem = copy.deepcopy(elem)
            new_elem.name = get_new_name(new_elem.name)

            # Prefix the symbolic value for one-port and two-port elements
            # (skip sensors whose value is empty/unused)
            if isinstance(new_elem, ElementOnePort):
                old_val = new_elem.value
                if str(old_val):  # skip empty-value sensors
                    new_elem.value = sp.Symbol(
                        get_new_name(str(old_val)),
                        real=True, positive=True,
                    )
                new_elem.bond = None  # will be re-wired by parent

            elif isinstance(new_elem, ElementTwoPort):
                old_val = new_elem.value
                new_elem.value = sp.Symbol(
                    get_new_name(str(old_val)),
                    real=True, positive=True,
                )
                new_elem.bond1 = None
                new_elem.bond2 = None

            elif isinstance(new_elem, Junction):
                new_elem.bonds = []
                new_elem.strong_bond = None

            elem_map[id(elem)] = new_elem

        # 2. Deep-copy bonds, re-pointing to new elements and prefixing symbols
        for bond in self.bondgraph.bonds:
            new_from = elem_map[id(bond.from_element)]
            new_to = elem_map[id(bond.to_element)]
            new_bond = Bond(
                from_element=new_from,
                to_element=new_to,
                causality=bond.causality,
                instance_name=instance_name,
                is_prefix=is_prefix
            )
            parent_bondgraph.add_bond(new_bond)

        # 3. Build new port mapping
        new_ports: Port = {}
        for port_name, node in self.ports.items():
            new_ports[port_name] = elem_map[id(node)]

        return new_ports

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, filepath: str | Path) -> None:
        """Save this sub-model definition to a JSON file.

        The file stores the structural description (element types,
        names, values, bond connectivity, causality, ports) — enough
        to reconstruct the sub-model from scratch.

        Parameters
        ----------
        filepath : str | Path
            Destination file path (typically ``*.json``).
        """
        data = self._to_dict()
        Path(filepath).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, filepath: str | Path) -> SubBondGraph:
        """Load a sub-model definition from a JSON file.

        Parameters
        ----------
        filepath : str | Path
            Source file path.

        Returns
        -------
        SubBondGraph
            The reconstructed sub-model.
        """
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        return cls._from_dict(data)

    # ------------------------------------------------------------------
    # Internal serialization helpers
    # ------------------------------------------------------------------

    def _to_dict(self) -> dict:
        """Convert this sub-model to a JSON-serializable dict."""
        from .elements import (
            SourceEffort, SourceFlow, Capacitor, Inductor,
            Resistor, Transformer, Gyrator,
            OneJunction, ZeroJunction,
        )
        from .sensors import IntegratedEffortSensor, IntegratedFlowSensor

        # Map each element to an index for bond connectivity
        elem_index: dict[int, int] = {}
        elements_data = []
        for i, elem in enumerate(self.bondgraph.elements):
            elem_index[id(elem)] = i
            entry = {
                "type": type(elem).__name__,
                "name": elem.name,
            }
            if isinstance(elem, (ElementOnePort, ElementTwoPort)):
                if not isinstance(elem, (IntegratedEffortSensor, IntegratedFlowSensor)):
                    entry["value"] = str(elem.value)
            elements_data.append(entry)

        bonds_data = []
        for bond in self.bondgraph.bonds:
            bonds_data.append({
                "from": elem_index[id(bond.from_element)],
                "to": elem_index[id(bond.to_element)],
                "causality": bond.causality.value,
            })

        ports_data = {}
        for port_name, node in self.ports.items():
            ports_data[port_name] = {
                "node_index": elem_index[id(node)],
            }

        return {
            "name": self.name,
            "elements": elements_data,
            "bonds": bonds_data,
            "ports": ports_data,
        }

    @classmethod
    def _from_dict(cls, data: dict) -> SubBondGraph:
        """Reconstruct a SubBondGraph from a dict (inverse of ``_to_dict``)."""
        from .bondgraph import BondGraph
        from .elements import (
            SourceEffort, SourceFlow, Capacitor, Inductor,
            Resistor, Transformer, Gyrator,
            OneJunction, ZeroJunction,
        )
        from .sensors import IntegratedEffortSensor, IntegratedFlowSensor

        # Element type registry
        TYPE_MAP = {
            "SourceEffort": SourceEffort,
            "SourceFlow": SourceFlow,
            "Capacitor": Capacitor,
            "Compliance": Capacitor,
            "Inductor": Inductor,
            "Inertance": Inductor,
            "Resistor": Resistor,
            "Resistance": Resistor,
            "Transformer": Transformer,
            "Gyrator": Gyrator,
            "OneJunction": OneJunction,
            "ZeroJunction": ZeroJunction,
            "IntegratedEffortSensor": IntegratedEffortSensor,
            "IntegratedFlowSensor": IntegratedFlowSensor,
        }

        # Reconstruct elements
        elements = []
        for entry in data["elements"]:
            elem_cls = TYPE_MAP[entry["type"]]
            if issubclass(elem_cls, Junction):
                elem = elem_cls(name=entry["name"])
            elif issubclass(elem_cls, (IntegratedEffortSensor, IntegratedFlowSensor)):
                elem = elem_cls(name=entry["name"])
            else:
                elem = elem_cls(name=entry["name"], value=entry["value"])
            elements.append(elem)

        # Reconstruct bonds and build BondGraph
        bg = BondGraph(name=data["name"])
        for bond_entry in data["bonds"]:
            bond = Bond(
                from_element=elements[bond_entry["from"]],
                to_element=elements[bond_entry["to"]],
                causality=bond_entry["causality"],
            )
            bg.add_bond(bond)

        # Reconstruct ports
        ports: Port = {}
        for port_name, port_data in data["ports"].items():
            # Support both old "junction_index" and new "node_index" keys
            idx = port_data.get("node_index", port_data.get("junction_index"))
            ports[port_name] = elements[idx]

        return cls(name=data["name"], bondgraph=bg, ports=ports)

    def __repr__(self) -> str:
        port_names = list(self.ports.keys())
        return (
            f"SubBondGraph(name={self.name!r}, "
            f"elements={len(self.bondgraph.elements)}, "
            f"bonds={len(self.bondgraph.bonds)}, "
            f"ports={port_names})"
        )
