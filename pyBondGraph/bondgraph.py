from __future__ import annotations

from pathlib import Path

import sympy as sp
import networkx as nx
import numpy as np

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from collections.abc import Callable
from typing import TYPE_CHECKING

from .core import Causality, CausalityError, DerivativeCausalityError, Node, StatefulElement, Bond, ElementOnePort, ElementTwoPort, Junction
from .elements import SourceEffort, SourceFlow, Capacitor, Inductor, Resistor, Transformer, Gyrator, OneJunction, ZeroJunction

from .core import Port
from .numerics import to_numpy, to_control_ss
from .fmu_export import to_fmu

if TYPE_CHECKING:
    from .subbondgraph import SubBondGraph

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

        Iterates over each bond and calls _handle_element() for both the from_element and to_element
        """

        # Reset element bond references so that repeated calls are safe
        for el in self.elements:
            if isinstance(el, ElementOnePort):
                el.bond = None
            elif isinstance(el, ElementTwoPort):
                el.bond1 = None
                el.bond2 = None
            elif isinstance(el, Junction):
                el.bonds = []
                el.strong_bond = None
        

        def _handle_element(element: Node, bond: Bond) -> None:
            """Assign bond to element according to the element's type."""
            if isinstance(element, ElementOnePort):
                element.bond = bond

            elif isinstance(element, ElementTwoPort):
                # bond1 = bond INTO the element, bond2 = bond FROM the element
                if bond.to_element is element:
                    element.bond1 = bond
                elif bond.from_element is element:
                    element.bond2 = bond
                else:
                    raise ValueError(f"Bond {bond} is not connected to ElementTwoPort {element}. You have called this function incorrectly.")

            elif isinstance(element, Junction):
                if bond not in element.bonds:
                    element.bonds.append(bond)

                if self._is_strong_for(bond, element):
                    if element.strong_bond is None:
                        element.strong_bond = bond
                    else:
                        raise ValueError(
                            f"{type(element).__name__} {element} already has a strong bond: "
                            f"{element.strong_bond}. Cannot assign {bond}."
                        )

        for bond in self.bonds:
            _handle_element(bond.from_element, bond)
            _handle_element(bond.to_element, bond)

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
                    raise ValueError(f"Element {element} is not fully connected with bonds.")

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

        # Auto-assign causality if any bonds lack it
        if any(b.causality is None for b in self.bonds):
            self.assign_causality()

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

    def get_substitution_dict(self, overrides: dict[sp.Symbol, float] | None = None) -> dict[sp.Symbol, float]:
        """Build a substitution dictionary from elements that have a ``numeric_value`` set.

        Iterates over all :class:`ElementOnePort` and :class:`ElementTwoPort` elements
        in the bond graph and maps each element's symbolic ``value`` to its ``numeric_value``.
        Elements without a ``numeric_value`` (i.e. ``None``) are silently skipped.

        Parameters
        ----------
        overrides : dict[sp.Symbol, float] | None, optional
            Additional or overriding entries merged into the result.
            This is useful for parameter sweeps or for supplying values
            that are not stored on the elements (e.g. external inputs).

        Returns
        -------
        dict[sp.Symbol, float]
            Mapping from symbolic parameter to numeric value.

        Raises
        ------
        ValueError
            If any element with a ``numeric_value`` would overwrite an
            already-collected symbol with a *different* value (duplicate
            symbols with the same numeric value are fine).
        """
        subs: dict[sp.Symbol, float] = {}

        for elem in self.elements:

            if isinstance(elem, (ElementOnePort, ElementTwoPort)) and elem.numeric_value is not None:
                if elem.value in subs and subs[elem.value] != elem.numeric_value:
                    raise ValueError(
                        f"Conflicting numeric values for symbol '{elem.value}': "
                        f"{subs[elem.value]} vs {elem.numeric_value} (element '{elem.name}')."
                    )
                
                subs[elem.value] = elem.numeric_value

        # update dict with manually provded overrides (if any)
        if overrides:
            subs.update(overrides)

        return subs

    def get_numeric_state_space(
        self,
        subs: dict[sp.Symbol, float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the numeric (numpy) state-space matrices ``(A, B, C, D)``.

        This is a convenience wrapper around :meth:`get_state_space` that
        substitutes numeric parameter values and converts the resulting
        symbolic matrices to :class:`numpy.ndarray`.

        Parameters
        ----------
        subs : dict[sp.Symbol, float] | None, optional
            Explicit substitution dictionary.  If ``None``,
            :meth:`get_substitution_dict` is called to collect the values
            stored on the elements.  If provided, it is used as-is (no
            merging with element values).

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Numeric matrices ``(A, B, C, D)``.

        Raises
        ------
        ValueError
            If the symbolic state-space cannot be computed, or if free
            symbols remain after substitution (i.e. some parameters have
            no numeric value).
        """
        A, B, C, D, _x, _ns, _ni, _no = self.get_state_space()

        if subs is None:
            subs = self.get_substitution_dict()

        # Check for remaining free symbols before conversion
        free = set()
        for M in (A, B, C, D):
            free |= M.free_symbols
        remaining = free - set(subs.keys())

        if remaining:
            raise ValueError(
                f"The following symbols have no numeric value: {remaining}. "
                f"Set numeric_value on the corresponding elements or pass them via the subs parameter."
            )

        return tuple(to_numpy(M, subs) for M in (A, B, C, D))
    
    def to_control_ss(self):
        """Generate a control.StateSpace object from the bond graph's numeric state-space matrices.

        Returns
        -------
        control.StateSpace
            The :class:`control.StateSpace` object representing the bond graph's linear dynamics.

        Raises
        ------
        ValueError
            Raises a value error through :meth:`get_numeric_state_space` if not all numerical values required are set beforehand in the bondgraph elements            
        """

        matrices = self.get_numeric_state_space()
        return to_control_ss(*matrices)



    def to_fmu(self, model_name: str = "BondGraphExport", dest: str | Path = ".", author_name: str = "MtP", description: str = "A bond graph model.", keep_slave_python_code: bool = False) -> None:
        # if default class name is used, use the bondgraph name if it is set
        if model_name == "BondGraphExport" and self.name:
            model_name = self.name

        to_fmu(self, dest=dest, class_name=model_name, author_name=author_name, description=description, keep_slave_python_code=keep_slave_python_code)


    def add_subbondgraph(self, sub_bondgraph: SubBondGraph, instance_name: str | None = None, is_prefix: bool = True) -> Port:
        """Instantiate a SubBondGraph into this bond graph.

        Deep-copies all elements and bonds from the sub-model with
        namespace prefixing, merges them into this graph, and returns
        the instantiated ports for subsequent ``connect()`` calls.

        Parameters
        ----------
        sub_bondgraph : SubBondGraph
            The sub-model to instantiate.
        instance_name : str | None, optional
            Override the namespace prefix.  Defaults to the sub-model's name.
        is_prefix : bool, optional
            Whether to use the instance name as a prefix or suffix for the copied elements.  Defaults to True.

        Returns
        -------
        Port
            Mapping of port names to new Node objects belonging
            to this graph.
        """
        return sub_bondgraph._instantiate(self, instance_name, is_prefix)

    def connect(
        self,
        node_a: Node,
        node_b: Node,
        causality: Causality | None = None,
    ) -> Bond:
        """Connect two nodes by adding a bond between them.

        Also used after :meth:`add_subbondgraph` to wire up the
        returned port nodes — but any :class:`Node` already in the
        graph (or about to be added) works.

        ``node_a`` becomes the bond's ``from_element`` and ``node_b``
        the ``to_element`` — just like the positional convention of
        the :class:`Bond` constructor.  For two-port elements
        (:class:`Transformer`, :class:`Gyrator`) the existing
        ``__handle_bonds`` logic uses this direction to assign
        ``bond1`` (to_element) vs ``bond2`` (from_element), so no
        extra ``side`` parameter is needed::

            # Bond INTO gyrator → bond1 (primary)
            system.connect(junction_node, gyrator, causality)

            # Bond FROM gyrator → bond2 (secondary)
            system.connect(gyrator, junction_node, causality)

        Parameters
        ----------
        node_a : Node
            Source — becomes ``from_element`` of the new bond.
        node_b : Node
            Destination — becomes ``to_element`` of the new bond.
        causality : Causality | None, optional
            Causality of the connecting bond.  Defaults to ``None``,
            meaning causality will be assigned later by
            :meth:`assign_causality` (SCAP).  Pass a ``Causality``
            value explicitly to override automatic assignment.

        Returns
        -------
        Bond
            The newly created connecting bond.

        Raises
        ------
        TypeError
            If an argument is not a Node.
        """
        if not isinstance(node_a, Node):
            raise TypeError(
                f"node_a must be a Node, got {type(node_a).__name__}"
            )
        if not isinstance(node_b, Node):
            raise TypeError(
                f"node_b must be a Node, got {type(node_b).__name__}"
            )

        bond = Bond(
            from_element=node_a,
            to_element=node_b,
            causality=causality,
        )
        self.add_bond(bond)
        return bond

    # --- shared helpers (used by both assign_causality and __handle_bonds) ---
    def _elements_of_type(self, *types: type) -> list[Node]:
        """Yield all elements that are instances of types."""
        return [el for el in self.elements if isinstance(el, types)]

    @staticmethod
    def _is_strong_for(bond: Bond, junc: Junction) -> bool:
        """Return True if bond is the strong bond at junc."""
        if bond.causality is None:
            return False
        
        if isinstance(junc, OneJunction):
            return (
                (bond.from_element is junc and bond.causality == Causality.EFFORT_OUT)
                or (bond.to_element is junc and bond.causality == Causality.FLOW_OUT)
            )
        elif isinstance(junc, ZeroJunction):
            return (
                (bond.from_element is junc and bond.causality == Causality.FLOW_OUT)
                or (bond.to_element is junc and bond.causality == Causality.EFFORT_OUT)
            )
        return False

    @staticmethod
    def _make_strong(bond: Bond, junc: Junction) -> Causality:
        """Return the causality that makes bond the strong bond at junc."""
        junc_is_from = bond.from_element is junc

        if isinstance(junc, OneJunction):
            return Causality.EFFORT_OUT if junc_is_from else Causality.FLOW_OUT
        else:  # ZeroJunction
            return Causality.FLOW_OUT if junc_is_from else Causality.EFFORT_OUT

    @staticmethod
    def _make_weak(bond: Bond, junc: Junction) -> Causality:
        """Return the causality that makes bond a weak bond at junc.

        This is the inverse of :meth:`_make_strong`.
        """
        # inverse mapping of strong to weak causality
        strong_to_weak = {
            Causality.EFFORT_OUT: Causality.FLOW_OUT,
            Causality.FLOW_OUT: Causality.EFFORT_OUT,
        }

        strong = BondGraph._make_strong(bond, junc)

        return strong_to_weak[strong]

    def assign_causality(self) -> None:
        """Run the Sequential Causality Assignment Procedure (SCAP).

        Assigns causality to every bond that currently has
        ``causality = None``.  Bonds whose causality was set explicitly
        (manually or via ``add_bond`` / ``connect``) are respected as
        fixed constraints.

        The algorithm proceeds in priority order:

        1. Sources: fixed causality (Se outputs effort, Sf outputs flow).
        2. Storage elements: preferred integral causality (C outputs effort, I outputs flow).
        3. Remaining bonds: assigned by propagation through junctions and two-port elements.

        After each assignment the consequences are propagated through connected junctions
        (exactly one strong bond each) and two-port elements
        (transformer: same causality on both bonds; gyrator: opposite causality).

        Raises
        ------
        DerivativeCausalityError
            If a storage element cannot receive its preferred integral causality.
            This indicates the system contains algebraic constraints (DAE instead of ODE) which are not supported.
        CausalityError
            If causality cannot be fully resolved (should not happen for a well-formed bond graph).
        """

        # Populate bond references on elements so we can use access them when assigning causality.
        self.__handle_bonds()

        # Track which bonds still need assignment
        unassigned: set[Bond] = {b for b in self.bonds if b.causality is None}

        if not unassigned:
            return

        def _assign(bond: Bond, causality: Causality) -> None:
            """Assign causality to a bond and remove it from the unassigned set."""
            bond.causality = causality
            unassigned.discard(bond)

        def _propagate() -> None:
            """Propagate causality through junctions and two-port elements
            until no further assignments can be made."""
            if not unassigned:
                return

            changed = True
            while changed:
                changed = False

                # Junctions: if any strong bond exists, all unassigned bonds must be weak
                for junc in self._elements_of_type(Junction):
                    assigned_bonds = [b for b in junc.bonds if b.causality is not None]
                    unassigned_here = [b for b in junc.bonds if b.causality is None]
                    if not unassigned_here:
                        continue

                    has_strong = any(self._is_strong_for(b, junc) for b in assigned_bonds)

                    if has_strong:
                        for b in unassigned_here:
                            _assign(b, self._make_weak(b, junc))
                            changed = True
                    elif len(unassigned_here) == 1:
                        _assign(unassigned_here[0], self._make_strong(unassigned_here[0], junc))
                        changed = True

                # Two-port elements: if one bond has causality, the other must be assigned accordingly
                for tp_elem in self._elements_of_type(ElementTwoPort):
                    b1, b2 = tp_elem.bond1, tp_elem.bond2
                    if b1 is None or b2 is None:
                        continue

                    src, dst = None, None
                    if b1.causality is not None and b2.causality is None:
                        src, dst = b1, b2
                    elif b2.causality is not None and b1.causality is None:
                        src, dst = b2, b1
                    else:
                        continue

                    if isinstance(tp_elem, Transformer):
                        _assign(dst, src.causality)
                    else:  # Gyrator — opposite causality
                        opp = Causality.FLOW_OUT if src.causality == Causality.EFFORT_OUT else Causality.EFFORT_OUT
                        _assign(dst, opp)
                    changed = True

                if not unassigned:
                    return

        def _desired_causality(element: ElementOnePort, bond: Bond) -> Causality:
            """Return the causality the element wants (from from_element perspective)."""
            el_is_from = bond.from_element is element
            if isinstance(element, (SourceEffort, Capacitor)):
                return Causality.EFFORT_OUT if el_is_from else Causality.FLOW_OUT
            elif isinstance(element, (SourceFlow, Inductor)):
                return Causality.FLOW_OUT if el_is_from else Causality.EFFORT_OUT
            return None

        def _would_conflict_with_junction(bond: Bond, causality: Causality) -> bool:
            """Check if assigning *causality* to *bond* would create a
            second strong bond at any connected junction."""
            for el in bond.elements:
                if not isinstance(el, Junction):
                    continue
                # Temporarily check what this would mean
                old = bond.causality
                bond.causality = causality
                is_strong = self._is_strong_for(bond, el)
                bond.causality = old
                if is_strong:
                    other_bonds = [b for b in el.bonds if b is not bond and b.causality is not None]
                    if any(self._is_strong_for(ob, el) for ob in other_bonds):
                        return True
            return False

        # --- Phase 1: Sources (fixed causality) -----------------------------
        for elem in self._elements_of_type(SourceEffort, SourceFlow):
            if elem.bond.causality is None:
                _assign(elem.bond, _desired_causality(elem, elem.bond))
        _propagate()

        # --- Phase 2: Storage elements (integral causality) -----------------
        for elem in self._elements_of_type(Capacitor, Inductor):
            if elem.bond.causality is not None:
                continue
            desired = _desired_causality(elem, elem.bond)
            if not _would_conflict_with_junction(elem.bond, desired):
                _assign(elem.bond, desired)
            else:
                raise DerivativeCausalityError(
                    f"Storage element '{elem.name}' ({type(elem).__name__}) "
                    f"cannot receive its preferred integral causality "
                    f"because a connected junction already has a strong bond. "
                    f"This forces derivative causality, turning the system "
                    f"into a DAE (differential-algebraic equation) which is "
                    f"not supported.\n"
                    f"Suggestion: insert a resistive element (R) between "
                    f"conflicting storage elements, or restructure the model."
                )
        _propagate()

        # --- Phase 3: Resistors and remaining bonds -------------------------
        for elem in self._elements_of_type(Resistor):
            if elem.bond.causality is not None:
                continue
            bond = elem.bond
            # Determine from the connected junction what this bond needs
            other_el = bond.to_element if bond.from_element is elem else bond.from_element
            if isinstance(other_el, Junction):
                assigned_bonds = [b for b in other_el.bonds if b.causality is not None]
                has_strong = any(self._is_strong_for(b, other_el) for b in assigned_bonds)
                if has_strong:
                    _assign(bond, self._make_weak(bond, other_el))
                elif len([b for b in other_el.bonds if b.causality is None]) == 1:
                    # if this is the only unassigned bond at the junction, it must be strong
                    _assign(bond, self._make_strong(bond, other_el))
        _propagate()

        # --- Validate: two-port causality consistency -----------------------
        for tp_elem in self._elements_of_type(ElementTwoPort):
            b1, b2 = tp_elem.bond1, tp_elem.bond2
            if b1 is None or b2 is None:
                continue
            if isinstance(tp_elem, Transformer) and b1.causality != b2.causality:
                raise CausalityError(
                    f"Transformer '{tp_elem.name}' requires both bonds to have the "
                    f"same causality, but bond1={b1.causality} and bond2={b2.causality}."
                )
            if isinstance(tp_elem, Gyrator) and b1.causality == b2.causality:
                raise CausalityError(
                    f"Gyrator '{tp_elem.name}' requires both bonds to have different "
                    f"causality, but both are {b1.causality}."
                )

        # --- Validate: unassigned bonds -------------------------------------
        if unassigned:
            descriptions = [f"  Bond({b.from_element.name} -> {b.to_element.name})" for b in unassigned]
            raise CausalityError(
                f"SCAP could not assign causality to {len(unassigned)} bond(s):\n"
                + "\n".join(descriptions)
                + "\nCheck that the bond graph is fully connected and well-formed."
            )

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
                bond=bond,
            )

        # Try Graphviz 'dot' layout first — it minimises edge crossings
        # and produces clean left-to-right hierarchical layouts.
        try:
            pos = nx.drawing.nx_agraph.graphviz_layout(
                G, prog="dot",
                args='-Grankdir=LR -Gnodesep=0.8 -Granksep=1.2 -Gordering=out'
            )
        except (ImportError, Exception):
            # Fall back to the previous layout strategy
            if nx.is_directed_acyclic_graph(G):
                for layer, nodes in enumerate(nx.topological_generations(G)):
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
        # Draw bond-number labels at varied positions along each edge so
        # that crossing bonds don't produce overlapping text.
        for edge_idx, (u, v, data) in enumerate(G.edges(data=True)):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            # Vary the parameter t \in [0.35, 0.65] per edge
            t = 0.35 + 0.3 * ((edge_idx * 7 + 3) % 11) / 10.0
            lx = x1 + t * (x2 - x1)
            ly = y1 + t * (y2 - y1)
            ax.text(
                lx, ly, str(data.get("label", "")),
                fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
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

            # Convert node size (points^2, i.e. area) to radius in pixels
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

        # Collect causal-stroke specs so they can be redrawn on resize.
        # One entry per bond
        _stroke_specs: list[tuple] = []
        for u, v, data in G.edges(data=True):
            causality = data.get("causality", None)
            if causality is Causality.EFFORT_OUT:
                _stroke_specs.append((pos[u], pos[v], "head", -2))
            elif causality is Causality.FLOW_OUT:
                _stroke_specs.append((pos[u], pos[v], "tail", -2))
            elif causality is None:
                pass  # No causality assigned yet — skip causal stroke
            else:
                raise ValueError(f"Edge {u}->{v} has no valid causality: {causality} --> this should never happen!")

        _stroke_artists: list[Line2D] = []

        def _refresh_strokes(event=None):
            """Recompute causal strokes in current display coords."""

            for a in _stroke_artists:
                a.remove() #removes artist (Line2D) from axes
            _stroke_artists.clear()

            for p1, p2, at, padding in _stroke_specs:
                draw_causal_stroke(ax, p1, p2, at=at, padding=padding)

            # Newly created Line2D artists are the last len(_stroke_specs) items (i.e. # of bonds lines) on ax.lines --> store to remove later
            _stroke_artists.extend(ax.lines[-len(_stroke_specs):])

        _refresh_strokes()                                    # initial draw
        fig.canvas.mpl_connect("resize_event", _refresh_strokes)  # stay correct on resize

        return fig, ax
