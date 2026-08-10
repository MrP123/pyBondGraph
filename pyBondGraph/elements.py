from .core import Causality, StatefulElement, ElementOnePort, ElementTwoPort, Junction

import sympy as sp


class SourceEffort(ElementOnePort):
    """Represents a source of effort in the bond graph.
    A source of effort provides a constant effort to its port, the associated flow follows automatically."""

    def __init__(self, name: str, value: str):
        """Create a source of effort in the bond graph.

        Parameters
        ----------
        name : str
            The name of the element. Forwarded to the `Node` base class.
        value : str
            The name of the element value, is internally used for creating a `sympy.Symbol`.
        """
        super().__init__(name, value)

    @property
    def equations(self) -> list[sp.Expr]:
        # Source --> constant effort
        return [self.bond.effort - self.value]


class SourceFlow(ElementOnePort):
    """Represents a source of flow in the bond graph.
    A source of flow provides a constant flow to its port, the associated effort follows automatically.
    """

    def __init__(self, name: str, value: str):
        """Create a source of flow in the bond graph.

        Parameters
        ----------
        name : str
            The name of the element. Forwarded to the `Node` base class.
        value : str
            The name of the element value, is internally used for creating a `sympy.Symbol`.
        """
        super().__init__(name, value)

    @property
    def equations(self) -> list[sp.Expr]:
        # Source --> constant flow
        return [self.bond.flow - self.value]


class Capacitor(ElementOnePort, StatefulElement):
    """Represents a linear compliance/capacitance element in the bond graph.
    A capacitance relates the effort of its port with the integral of its flow by a constant capacitance value.
    """

    def __init__(self, name: str, value: str, numeric_value: float = None):
        """Create a linear compliance/capacitance element in the bond graph.

        Parameters
        ----------
        name : str
            The name of the element. Forwarded to the `Node` base class.
        value : str
            The name of the element value, is internally used for creating a `sympy.Symbol`.
        numeric_value : float, optional
            The numeric value of the capacitance, used for numerical simulations.
        """
        super().__init__(name, value, numeric_value)

    @property
    def state_var(self) -> sp.Symbol:
        """Returns the symbolic state variable (generalized displacement) associated with the compliance/capacitance.

        Returns
        -------
        sp.Symbol
            The symbolic state variable representing the generalized displacement (q) of the compliance/capacitance.
        """
        return sp.Symbol(f"q_{self.name}", real=True)

    @property
    def equations(self) -> list[sp.Expr]:
        """Returns the symbolic equations defining the behavior of the linear compliance/capacitance.
        Formulated in the derivative of the internal state_var, stemming from integral causality.

        Returns
        -------
        list[sp.Expr]
            A list of symbolic equations representing the behavior of the compliance/capacitance.
        """

        # Dynamic equation: f = dq/dt, e = q / C
        return [
            self.bond.flow - sp.Derivative(self.state_var, "t"),
            self.bond.effort - self.state_var / self.value,
        ]


type Compliance = Capacitor  # Alias for Compliance element


class Inductor(ElementOnePort, StatefulElement):
    """Represents a linear inertia/inductance element in the bond graph.
    An inductor relates the flow of its port with the integral of its effort by a constant inductance value.
    """

    def __init__(self, name: str, value: str, numeric_value: float = None):
        """Create a linear inertia/inductance element in the bond graph.

        Parameters
        ----------
        name : str
            The name of the element. Forwarded to the `Node` base class.
        value : str
            The name of the element value, is internally used for creating a `sympy.Symbol`.
        numeric_value : float, optional
            The numeric value of the inductance, used for numerical simulations.
        """
        super().__init__(name, value, numeric_value)

    @property
    def state_var(self) -> sp.Symbol:
        """Returns the symbolic state variable (generalized momentum) associated with the inertia/inductance.

        Returns
        -------
        sp.Symbol
            The symbolic state variable representing the generalized momentum (p) of the inertia/inductance.
        """
        return sp.Symbol(f"p_{self.name}", real=True)

    @property
    def equations(self) -> list[sp.Expr]:
        """Returns the symbolic equations defining the behavior of the linear inertia/inductance.
        Formulated in the derivative of the internal state_var, stemming from integral causality.

        Returns
        -------
        list[sp.Expr]
            A list of symbolic equations representing the behavior of the inertia/inductance.
        """

        # Dynamic equation: e = dp/dt, f = p / L
        return [
            self.bond.effort - sp.Derivative(self.state_var, "t"),
            self.bond.flow - self.state_var / self.value,
        ]


type Inertance = Inductor  # Alias for Inertia element


class Resistor(ElementOnePort):
    """Represents a linear resistance/damper element in the bond graph.
    A resistor relates the effort and flow of its port by a constant resistance value.
    """

    def __init__(self, name: str, value: str, numeric_value: float = None):
        """Create a linear resistance element.

        Parameters
        ----------
        name : str
            The name of the resistance. Forwarded to the `Node` base class.
        value : str
            The name of the resistance value, is internally used for creating a `sympy.Symbol`.
        numeric_value : float, optional
            The numeric value of the resistance, used for numerical simulations.
        """
        super().__init__(name, value, numeric_value)

    @property
    def equations(self) -> list[sp.Expr]:
        """Returns the symbolic equations defining the behavior of the linear resistance.
        The equation is formulated based on the causality of the associated bond.
        This makes resolving causality issues easier, as there is no preferred one for resistors.

        Returns
        -------
        list[sp.Expr]
            A list of symbolic equations representing the behavior of the resistance.
        """

        # Resistance: e = R * f
        if self.bond.causality == Causality.EFFORT_OUT:
            return [self.bond.flow - 1 / self.value * self.bond.effort]
        elif self.bond.causality == Causality.FLOW_OUT:
            return [self.bond.effort - self.value * self.bond.flow]


type Resistance = Resistor  # Alias for Resistance element


class Transformer(ElementTwoPort):
    """Represents a transformer element in the bond graph.
    A transformer relates the efforts of its two ports and the flows of its two ports by a constant ratio.
    """

    def __init__(self, name: str, value: str, numeric_value: float = None):
        """Create a transformer element in the bond graph.

        Parameters
        ----------
        name : str
            The name of the element. Forwarded to the `Node` base class.
        value : str
            The name of the element value, is internally used for creating a `sympy.Symbol`.
        numeric_value : float, optional
            The numeric value of the transformer ratio, used for numerical simulations.
        """
        super().__init__(name, value, numeric_value)

    @property
    def equations(self) -> list[sp.Expr]:
        """Returns the symbolic equations defining the behavior of the transformer.
        The equations are formulated based on the causality of the associated bonds.
        This makes resolving causality issues easier, as there is no preferred causality for transformers.

        Returns
        -------
        list[sp.Expr]
            A list of symbolic equations representing the behavior of the transformer.

        Raises
        ------
        ValueError
            If both bonds are not assigned or if they do not have the same causality.
        """

        if self.bond1 is None or self.bond2 is None:
            raise ValueError("Both bonds must be assigned to the transformer element.")

        if self.bond1.causality != self.bond2.causality:
            raise ValueError("Both bonds must have the same causality for a transformer element.")

        if self.bond1.causality == Causality.EFFORT_OUT:
            return [
                self.bond1.flow - 1 / self.value * self.bond2.flow,
                self.bond2.effort - 1 / self.value * self.bond1.effort,
            ]
        elif self.bond1.causality == Causality.FLOW_OUT:
            return [
                self.bond1.effort - self.value * self.bond2.effort,
                self.bond2.flow - self.value * self.bond1.flow,
            ]


class Gyrator(ElementTwoPort):
    """Represents a gyrator element in the bond graph.
    A gyrator relates the effort of one port with the flow of the other (and vice versa) by a constant ratio.
    """

    def __init__(self, name: str, value: str, numeric_value: float = None):
        """Create a gyrator element in the bond graph.

        Parameters
        ----------
        name : str
            The name of the element. Forwarded to the `Node` base class.
        value : str
            The name of the element value, is internally used for creating a `sympy.Symbol`.
        numeric_value : float, optional
            The numeric value of the gyrator ratio, used for numerical simulations.
        """
        super().__init__(name, value, numeric_value)

    @property
    def equations(self) -> list[sp.Expr]:
        """Returns the symbolic equations defining the behavior of the gyrator.
        The equations are formulated based on the causality of the associated bonds.
        This makes resolving causality issues easier, as there is no preferred causality for gyrators.

        Returns
        -------
        list[sp.Expr]
            A list of symbolic equations representing the behavior of the gyrator.

        Raises
        ------
        ValueError
            If both bonds are not assigned or if they have the same causality.
        """

        if self.bond1 is None or self.bond2 is None:
            raise ValueError("Both bonds must be assigned to the gyrator element.")

        if self.bond1.causality == self.bond2.causality:
            raise ValueError("Both bonds must have the different causality for a gyrator element.")

        if self.bond1.causality == Causality.EFFORT_OUT:
            return [
                self.bond1.flow - 1 / self.value * self.bond2.effort,
                self.bond2.flow - 1 / self.value * self.bond1.effort,
            ]
        elif self.bond1.causality == Causality.FLOW_OUT:
            return [
                self.bond1.effort - self.value * self.bond2.flow,
                self.bond2.effort - self.value * self.bond1.flow,
            ]


class OneJunction(Junction):
    """Represents a one-junction element in the bond graph.
    A one-junction enforces equal flow (current, velocity, etc.) on all connected bonds and the sum of efforts to be zero .
    The sign convention is that efforts of bonds going into the junction are positive, efforts of bonds going out of the junction are negative.
    """

    def __init__(self, name: str):
        """Create a one-junction element in the bond graph.

        Parameters
        ----------
        name : str
            The name of the element. Forwarded to the `Node` base class.
        """
        super().__init__(name)

    @property
    def equations(self) -> list[sp.Expr]:
        """Returns the symbolic equations defining the behavior of the one-junction.

        Returns
        -------
        list[sp.Expr]
            A list of symbolic equations representing the behavior of the one-junction.
        """

        effort_eq = 0
        for b in self.bonds:
            dir = +1 if b.to_element == self else -1
            effort_eq += dir * b.effort

        flow_eq = [self.bonds[0].flow - b.flow for b in self.bonds[1:]]
        return [effort_eq, *flow_eq]


class ZeroJunction(Junction):
    """Represents a zero-junction element in the bond graph.
    A zero-junction enforces equal effort (voltage, force, etc.) on all connected bonds and the sum of flows to be zero .
    The sign convention is that flows of bonds going into the junction are positive, flows of bonds going out of the junction are negative.
    """

    def __init__(self, name: str):
        """Create a zero-junction element in the bond graph.

        Parameters
        ----------
        name : str
            The name of the element. Forwarded to the `Node` base class.
        """
        super().__init__(name)

    @property
    def equations(self) -> list[sp.Expr]:
        """Returns the symbolic equations defining the behavior of the zero-junction.

        Returns
        -------
        list[sp.Expr]
            A list of symbolic equations representing the behavior of the zero-junction.
        """

        flow_eq = 0
        for b in self.bonds:
            dir = +1 if b.to_element == self else -1
            flow_eq += dir * b.flow

        effort_eq = [self.bonds[0].effort - b.effort for b in self.bonds[1:]]
        return [flow_eq, *effort_eq]
