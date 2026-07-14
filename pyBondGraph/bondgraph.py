from __future__ import annotations

import sympy as sp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Callable
from typing import TYPE_CHECKING

from .core import (
    Causality,
    Node,
    StatefulElement,
    Bond,
    ElementOnePort,
    ElementTwoPort,
    Junction,
)
from .elements import SourceEffort, SourceFlow, OneJunction, ZeroJunction

if TYPE_CHECKING:
    from .subbondgraph import SubBondGraph
    from .port import Port

type SolutionType = dict[sp.Expr, sp.Expr]


class BondGraph:
    """Represents a bond graph consisting of elements and the bonds that connect them."""

    def __init__(self, name: str = ""):
        """Initializes a new bond graph without any elements or bonds.

        Parameters
        ----------
        name : str, optional
            Name of the bond graph. Used as namespace prefix when merging sub-models.
        """

        self.name = name
        self.elements: list[Node] = []
        self.bonds: list[Bond] = []
        self._bond_counter = 0  # Instance-scoped bond counter

        self.state_vars: list[sp.Expr] = []
        self.equations: list[sp.Expr] = []
        self.inputs: list[sp.Symbol] = []

        self.solution: SolutionType = None

    def add_bond(self, bond: Bond) -> None:
        """Adds a bond to the bond graph by appending it to `self.bonds`.
        This adds the connected elements to `self.elements` if they are not already present.
        If the added bond connects to a `SourceEffort` or `SourceFlow`, its value is added to `self.inputs`.
        This is needed for generating the state space representation.

        Parameters
        ----------
        bond : Bond
            The bond to be added to the bond graph.

        Raises
        ------
        ValueError
            If the bond is already part of the bond graph.
        """

        if bond in self.bonds:
            raise ValueError(f"Bond {bond} is already part of the bond graph.")

        # Renumber the bond using the instance-scoped counter to avoid collisions
        subs = bond.rename_symbols(new_num=self._bond_counter)
        self._bond_counter += 1

        # Propagate renamed symbols into any equations that already reference the old symbols
        # (relevant when merging sub-models whose equations were built with different numbering)
        self.equations = [eq.subs(subs) for eq in self.equations]

        self.bonds.append(bond)

        for element in bond.elements:
            if element in self.elements:
                continue

            self.elements.append(element)

            if isinstance(element, SourceEffort) or isinstance(element, SourceFlow):
                self.inputs.append(element.value)

    def __handle_bonds(self) -> None:
        """Handles the bonds in the bond graph by assigning them to the appropriate elements.
        This assignment propagates the bond references to the elements, so that each element knows which bonds it is connected to.
        """

        def handle_bond_element(element: Node, bond: Bond):
            """Internal helper function to handle the assignment of a bond to an element.

            Parameters
            ----------
            element : Node
                Element of the bond to handle. Must be called with both `from_element` and `to_element` of the bond.
            bond : Bond
                The bond to assign to the element.

            Raises
            ------
            ValueError
                If the element is a junction and it is attemped to add a second strong bond.
            """

            if isinstance(element, ElementOnePort):
                element.bond = bond

            elif isinstance(element, ElementTwoPort):
                if bond.to_element == element:
                    element.bond1 = bond
                elif bond.from_element == element:
                    element.bond2 = bond

            elif isinstance(element, Junction):
                element.bonds.append(bond)

                if isinstance(element, OneJunction):
                    if (bond.from_element == element and bond.causality == Causality.EFFORT_OUT) or (bond.to_element == element and bond.causality == Causality.FLOW_OUT):
                        # bond is strong bond for one junction
                        if element.strong_bond is None:
                            element.strong_bond = bond
                            print(f"Assigned strong bond {bond} to OneJunction {element}.")
                        else:
                            raise ValueError(f"OneJunction {element} already has a strong bond: {element.strong_bond}. Cannot assign {bond}.")

                elif isinstance(element, ZeroJunction):
                    if (bond.from_element == element and bond.causality == Causality.FLOW_OUT) or (bond.to_element == element and bond.causality == Causality.EFFORT_OUT):
                        # bond is strong bond for zero junction
                        if element.strong_bond is None:
                            element.strong_bond = bond
                            print(f"Assigned strong bond {bond} to ZeroJunction {element}.")
                        else:
                            raise ValueError(f"ZeroJunction {element} already has a strong bond: {element.strong_bond}. Cannot assign {bond}.")

        # Call helper function for both elements of each bond
        for bond in self.bonds:
            handle_bond_element(bond.from_element, bond)
            handle_bond_element(bond.to_element, bond)

    def __handle_equations(self) -> None:
        """Accumulates the equations from all elements and junctions in the bond graph.
        Also collects the state variables from all stateful elements.

        Raises
        ------
        ValueError
            If an element is not fully connected with bonds.
        """

        for element in self.elements:
            if isinstance(element, ElementOnePort):
                bond = element.bond
                if bond is None:
                    raise ValueError(f"Element {element} has no connected bond.")

                # Add equations from the element to the bond graph
                self.equations.extend(element.equations)

                if isinstance(element, StatefulElement):
                    self.state_vars.append(element.state_var)

            elif isinstance(element, ElementTwoPort):
                bond1 = element.bond1
                bond2 = element.bond2
                if bond1 is None or bond2 is None:
                    raise ValueError(f"Element {element} has no connected bonds.")

                # Add equations from the element to the bond graph
                self.equations.extend(element.equations)

            elif isinstance(element, Junction):
                self.equations.extend(element.equations)

    def get_solution_equations(self) -> SolutionType:
        """Returns the symbolic equations defining the solution of the bond graph.
        The solution is computed by solving the accumulated equations of the bond graph symbolically using `sympy.solve`.

        Returns
        -------
        SolutionType
            A dictionary mapping each symbolic variable to its solved expression.
            The keys include the time derivatives of the state variables and the efforts and flows of all bonds.
        """

        self.__handle_bonds()
        self.__handle_equations()
        state_derivatives = [sp.Derivative(var, "t") for var in self.state_vars]
        self.solution = sp.solve(
            self.equations,
            state_derivatives
            + [b.effort for b in self.bonds]
            + [b.flow for b in self.bonds],
        )
        return self.solution

    def get_state_space(self) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, int, int, int]:
        """Calculates the linear state space representation of the bond graph.
        This method automatically calls `get_solution_equations` if the solution has not yet been computed.

        Depending on the number of state variables, inputs, and outputs, the state space representation is either SISO, MIMO or a hybrid.
        The general form of a (nonlinear) state space model is given by the equations:
            x_dot = f(x, u)
                y = h(x, u)
        where x is the state vector, u is the input vector, and y is the output vector.
        As the bond graph framework currently only supports linear elements, the state space representation can be simplified to a linear form.
        The linear state space representation is given by the matrices A, B, C, D in the equations:
            x_dot = A*x + B*u
                y = C*x + D*u
        The matrices are computed by taking the Jacobians of the functions f and h with respect to the state variables and inputs.
        The output y is defined to be all efforts and flows of all bonds in the bond graph, with the efforts coming first and then the flows.
        The input u is defined to be all sources (i.e. `SourceEffort` and `SourceFlow` elements) in the bond graph.
        The state variables x are defined to be the state variables of all `StatefulElement` elements in the bond graph.

        Returns
        -------
        tuple[sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, sp.Matrix, int, int, int]
            Returns the matrices A, B, C, D of the state space representation, the state vector x,
            as well as the number of states, inputs, and outputs.
            A in R^(n_states x n_states)
            B in R^(n_states x n_inputs)
            C in R^(n_outputs x n_states)
            D in R^(n_outputs x n_inputs)
            x in R^(n_states x 1)

        Raises
        ------
        ValueError
            If the system of equations for this bond graph could not be solved.
        """

        # Retrieve solution if needed
        if self.solution is None:
            if not self.get_solution_equations():
                raise ValueError("Could not compute solution.")

        n_states = len(self.state_vars)
        n_inputs = len(self.inputs)  # Number of inputs (sources)
        n_outputs = 2 * len(self.bonds)  # effort & flow for each bond

        # General form of a state space model
        # x_dot = f(x, u)
        #     y = h(x, u)
        # Simplification for linear systems:
        # x_dot = A*x + B*u
        #     y = C*x + D*u
        # --> therefore
        # A = ∂f/∂x, B = ∂f/∂u, C = ∂h/∂x, D = ∂h/∂u each at stationary point 0

        f: sp.Matrix = sp.zeros(n_states, 1)
        for i, state_var in enumerate(self.state_vars):
            state_deriv = sp.Derivative(state_var, "t")  # symbolic derivative dx/dt
            f[i] = self.solution[state_deriv]

        h: sp.Matrix = sp.zeros(n_outputs, 1)  # efforts then flows
        for i, bond in enumerate(self.bonds):
            h[i] = self.solution[bond.effort]
            h[i + n_outputs // 2] = self.solution[bond.flow]

        A = f.jacobian(self.state_vars)
        B = f.jacobian(self.inputs)

        C = h.jacobian(self.state_vars)
        D = h.jacobian(self.inputs)
        # alternatively could use sp.linear_eq_to_matrix(...)

        return A, B, C, D, sp.Matrix(self.state_vars), n_states, n_inputs, n_outputs

    def add_subbondgraph(self, sub_bondgraph: SubBondGraph, instance_name: str | None = None) -> dict:
        """Instantiate a SubBondGraph into this bond graph.

        Deep-copies all elements and bonds from the sub-model with
        namespace prefixing, merges them into this graph, and returns
        the instantiated ports for subsequent ``connect_ports()`` calls.

        Parameters
        ----------
        sub_bondgraph : SubBondGraph
            The sub-model to instantiate.
        instance_name : str | None, optional
            Override the namespace prefix.  Defaults to the sub-model's name.

        Returns
        -------
        dict[str, Port]
            Mapping of port names to new Port objects whose junctions
            belong to this graph.
        """
        return sub_bondgraph._instantiate(self, instance_name)

    def connect_ports(
        self,
        port_a: Port,
        port_b: Port,
        causality: Causality = Causality.EFFORT_OUT,
    ) -> Bond:
        """Connect two ports by adding a bond between their boundary junctions.

        Parameters
        ----------
        port_a : Port
            First port (bond ``from_element``).
        port_b : Port
            Second port (bond ``to_element``).
        causality : Causality, optional
            Causality of the connecting bond.  Defaults to
            ``Causality.EFFORT_OUT``.

        Returns
        -------
        Bond
            The newly created connecting bond.

        Raises
        ------
        ValueError
            If the ports belong to incompatible physical domains.
        """
        if (
            port_a.domain != "generic"
            and port_b.domain != "generic"
            and port_a.domain != port_b.domain
        ):
            raise ValueError(
                f"Cannot connect ports of different domains: "
                f"{port_a.domain!r} vs {port_b.domain!r}"
            )

        bond = Bond(
            from_element=port_a.junction,
            to_element=port_b.junction,
            causality=causality,
        )
        self.add_bond(bond)
        return bond

    def plot(self, layout: Callable[[nx.Graph, ...], dict] = nx.spectral_layout, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """Plots the bond graph as a `networkx` graph.

        Parameters
        ----------
        layout : Callable[[nx.Graph, ...], dict], optional
            `networkx` layout function for plotting the graph, by default `nx.spectral_layout`
            This is only called if the graph is not a directed acyclic graph (DAG).
            If the graph is a DAG, a `multipartite_layout` is used instead.

        Returns
        -------
        tuple[plt.Figure, plt.Axes]
            Matplotlib Figure and axes objects for the plot.
            
        Raises
        ------
        ValueError
            If an edge has no valid causality assigned.
        """

        G = nx.DiGraph()

        for elem in self.elements:
            G.add_node(elem.name, label=elem.name)

        for bond in self.bonds:
            G.add_edge(
                bond.from_element.name,
                bond.to_element.name,
                label=bond.num,
                causality=bond.causality,
            )

        # https://networkx.org/documentation/stable/auto_examples/graph/plot_dag_layout.html
        if nx.is_directed_acyclic_graph(G):
            for layer, nodes in enumerate(nx.topological_generations(G)):
                # `multipartite_layout` expects the layer as a node attribute, so add the
                # numeric layer value as a node attribute
                for node in nodes:
                    G.nodes[node]["layer"] = layer

            pos = nx.multipartite_layout(G, subset_key="layer")
        else:
            pos = layout(G, **kwargs)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_axis_off()
        nx.draw_networkx(
            G,
            pos,
            ax=ax,
            with_labels=True,
            node_size=2000,
            node_color="lightblue",
            font_size=10,
            font_color="black",
            arrows=True,
        )
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=nx.get_edge_attributes(G, "label"),
            font_size=8,
            ax=ax,
        )

        # Function to add a perpendicular line
        def draw_causal_stroke(ax, p1, p2, at="head", length=20, node_size=2000, padding=2):
            """Draw a causal stroke perpendicular to the edge (p1 -> p2).

            Parameters
            ----------
            ax : matplotlib.axes.Axes
                The axes to draw on.
            p1 : tuple[float, float]
                The (x, y) coordinates of the first node.
            p2 : tuple[float, float]
                The (x, y) coordinates of the second node.
            at : str, optional
                Where to draw the stroke, by default "head"
            length : int, optional
                The length of the stroke, by default 20
            node_size : int, optional
                The size of the nodes, by default 2000
            padding : int, optional
                The padding between the node and the stroke, by default 2

            Raises
            ------
            ValueError
                If the 'at' parameter is not "head" or "tail".
            """

            # --- Convert node size (points^2) to radius in pixels ---
            radius_points = np.sqrt(node_size / np.pi)
            radius_pixels = radius_points * ax.figure.dpi / 72.0  # 1 point = 1/72 inch
            offset = radius_pixels + padding   # + padding (in px) so stroke sits outside node

            # Transform points to display (pixel) coordinates
            p1_disp = ax.transData.transform(p1)
            p2_disp = ax.transData.transform(p2)

            # Edge vector in display coords
            vec = p2_disp - p1_disp
            norm = np.linalg.norm(vec)
            if norm == 0:
                return
            uvec = vec / norm

            # Perpendicular in display coords
            perp = np.array([-uvec[1], uvec[0]])

            # Base point (head or tail), offset in pixels
            if at == "head":
                base_disp = p2_disp - uvec * offset
            elif at == "tail":
                base_disp = p1_disp + uvec * offset
            else:
                raise ValueError(f"Invalid value for 'at': {at}")

            # Stroke endpoints in display coords
            p_start_disp = base_disp - perp * (length / 2)
            p_end_disp = base_disp + perp * (length / 2)

            # Transform back to data coords for plotting
            p_start = ax.transData.inverted().transform(p_start_disp)
            p_end = ax.transData.inverted().transform(p_end_disp)

            ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color="k", lw=1.0)

        # Add perpendicular lines to edges
        for u, v, data in G.edges(data=True):
            causality = data.get("causality", None)

            if causality is Causality.EFFORT_OUT:
                draw_causal_stroke(ax, pos[u], pos[v], at="head", padding=-2)
            elif causality is Causality.FLOW_OUT:
                draw_causal_stroke(ax, pos[u], pos[v], at="tail", padding=-2)
            else:
                raise ValueError(f"Edge {u}->{v} has no valid causality: {causality} --> this should never happen!")

        return fig, ax
