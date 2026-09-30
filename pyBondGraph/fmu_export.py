from pythonfmu3 import Fmi3Causality, Fmi3SlaveBase, Fmi3Variability, Fmi3Initial, Float64, ModelExchange
from typing import List

from pyBondGraph import BondGraph, Causality, SourceEffort, Inductor, Resistor, OneJunction, Gyrator
import sympy as sp


class BondGraphSlave(Fmi3SlaveBase, ModelExchange):
    """Generic Model-Exchange FMU wrapper for an arbitrary (linear) bond graph.

    The only model-specific part is :meth:`build_bond_graph`. Everything else
    (state/derivative/output/parameter registration and the runtime evaluation
    of the state-space equations) is derived automatically from the bond graph.

    Model parameters are kept *symbolic* and the state-space matrices are
    evaluated at runtime from the current parameter attribute values via
    ``sympy.lambdify``. This means changing a parameter from the outside
    (e.g. from Simulink) actually changes the dynamics, instead of the values
    being baked in once at export time.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.author = "MtP"
        self.description = "Bondgraph test"

        # ----- model-specific part -------------------------------------------
        bond_graph = self.build_bond_graph()
        bond_graph.assign_causality()

        # ----- everything below is generic -----------------------------------
        self._build_from_bondgraph(bond_graph)

    # -------------------------------------------------------------------------
    # Model definition (the only hardcoded part)
    # -------------------------------------------------------------------------
    def build_bond_graph(self) -> BondGraph:
        """Build and return the bond graph to be exported.

        Swap the body of this method to export a different model; the rest of
        the class is fully generic.
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

    # -------------------------------------------------------------------------
    # Generic setup
    # -------------------------------------------------------------------------
    def _build_from_bondgraph(self, bond_graph: BondGraph) -> None:
        self.bg = bond_graph

        # Symbolic state-space. Keep the symbolic matrices so parameters stay live.
        A, B, C, D, _x, n_states, n_inputs, n_outputs = bond_graph.get_state_space()
        self.A, self.B, self.C, self.D = A, B, C, D
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        # n_outputs = 2 * n_bonds; efforts occupy rows [0, n_bonds), flows [n_bonds, 2*n_bonds).
        # This effort-then-flow stride is baked into the C and D matrices by get_state_space().
        self.n_bonds = len(bond_graph.bonds)

        # Free parameters = every symbol in A/B/C/D that is not an input.
        free_vars = set()
        for M in (A, B, C, D):
            free_vars |= M.free_symbols
        for inp in bond_graph.inputs:
            free_vars.discard(inp)
        # Deterministic order for lambdify argument lists and reproducibility.
        self._params = sorted(free_vars, key=lambda s: s.name)

        # Lambdify to pure-python callables (no numpy dependency in the FMU sandbox).
        # Using .tolist() keeps the output as nested Python lists evaluated with `math`.
        arglist = self._params
        self._A_fn = sp.lambdify(arglist, A.tolist(), modules="math")
        self._B_fn = sp.lambdify(arglist, B.tolist(), modules="math")
        self._C_fn = sp.lambdify(arglist, C.tolist(), modules="math")
        self._D_fn = sp.lambdify(arglist, D.tolist(), modules="math")

        subs = bond_graph.get_substitution_dict()

        # ----- register FMU variables ----------------------------------------
        # Helper: create a Float64 with EXPLICIT getter/setter bound to an
        # instance attribute. This bypasses pythonfmu3's fragile auto-setter
        # logic (fmi3slave.py: register_variable), which only creates a setter
        # if hasattr(self, name) is already True at registration time AND uses a
        # name that has been demangled via lstrip('_') (variables.py). Relying on
        # that ordering left the e_i/f_i outputs with setter=None, so the first
        # fmi3SetFloat64 on them crashed with "'NoneType' object is not callable".
        def make_var(name: str, start: float = 0.0, **kwargs) -> Float64:
            setattr(self, name, float(start))
            var = Float64(name, **kwargs)
            var.getter = (lambda n=name: getattr(self, n))
            var.setter = (lambda v, n=name: setattr(self, n, v))
            return var

        self.time = 0.0
        self.register_variable(
            make_var("time", causality=Fmi3Causality.independent, variability=Fmi3Variability.continuous)
        )

        # States (outputs) + their derivatives (local). The FMI `derivative`
        # attribute must reference the value reference of the state it derives.
        for state_var in bond_graph.state_vars:
            # NOTE: register_variable() returns None; it assigns the value
            # reference *onto the variable object*. We must read it back from
            # there, otherwise `derivative=None` and pythonfmu3 counts zero
            # continuous states (breaking integration and crashing Simulink).
            # IMPORTANT: the instance attribute must exist BEFORE register_variable
            # is called. pythonfmu3 only auto-creates a setter when
            # hasattr(self, name) is already True at registration time
            # (see fmi3slave.py: register_variable). Registering first and
            # setattr-ing afterwards leaves setter=None, which makes any
            # fmi3SetFloat64 call crash with "'NoneType' object is not callable".
            setattr(self, state_var.name, 0.0)
            setattr(self, f"der_{state_var.name}", 0.0)
            state = Float64(
                state_var.name,
                causality=Fmi3Causality.output,
                start=0,
                variability=Fmi3Variability.continuous,
                initial=Fmi3Initial.exact,
            )
            self.register_variable(state)
            self.register_variable(
                Float64(
                    f"der_{state_var.name}",
                    causality=Fmi3Causality.local,
                    variability=Fmi3Variability.continuous,
                    derivative=state.value_reference,
                )
            )

        # Inputs (sources). The attribute must exist BEFORE register_variable so
        # pythonfmu3 auto-creates a setter (see states above). Without this the
        # first fmi3SetFloat64 on the input crashes with a NoneType setter.
        for input_var in bond_graph.inputs:
            setattr(self, input_var.name, 0.0)
            self.register_variable(
                Float64(
                    input_var.name,
                    causality=Fmi3Causality.input,
                    variability=Fmi3Variability.continuous,
                    initial=Fmi3Initial.exact,
                )
            )

        # Outputs: effort of every bond, then flow of every bond.
        for i in range(self.n_bonds):
            self.register_variable(
                Float64(
                    f"e_{i + 1}",
                    causality=Fmi3Causality.output,
                    variability=Fmi3Variability.continuous,
                    initial=Fmi3Initial.exact,
                )
            )
            setattr(self, f"e_{i + 1}", 0.0)

        for i in range(self.n_bonds):
            self.register_variable(
                Float64(
                    f"f_{i + 1}",
                    causality=Fmi3Causality.output,
                    variability=Fmi3Variability.continuous,
                    initial=Fmi3Initial.exact,
                )
            )
            setattr(self, f"f_{i + 1}", 0.0)

        # Parameters: configurable from the outside (e.g. Simulink), start from
        # the numeric values stored on the elements where available.
        # IMPORTANT: the attribute must exist BEFORE register_variable, exactly
        # like states/inputs above. Otherwise pythonfmu3 leaves setter=None
        # (see fmi3slave.py: register_variable) and the first fmi3SetFloat64 on
        # a parameter crashes Simulink with "'NoneType' object is not callable".
        for var in self._params:
            setattr(self, var.name, float(subs.get(var, 0.0)))
            self.register_variable(
                Float64(
                    var.name,
                    causality=Fmi3Causality.parameter,
                    variability=Fmi3Variability.fixed,
                    initial=Fmi3Initial.exact,
                )
            )

    # -------------------------------------------------------------------------
    # Runtime evaluation
    # -------------------------------------------------------------------------
    @staticmethod
    def _mat_vec(M, x):
        """Matrix (nested list) times vector, pure python."""
        return [sum(M[i][j] * x[j] for j in range(len(x))) for i in range(len(M))]

    def _current_matrices(self):
        """Evaluate the symbolic state-space at the current parameter values."""
        pvals = [getattr(self, p.name) for p in self._params]
        A = self._A_fn(*pvals)
        B = self._B_fn(*pvals)
        C = self._C_fn(*pvals)
        D = self._D_fn(*pvals)
        return A, B, C, D

    def _state_and_input_vectors(self):
        x = [getattr(self, sv.name) for sv in self.bg.state_vars]
        u = [getattr(self, iv.name) for iv in self.bg.inputs]
        return x, u

    def _update_outputs(self, C, D, x, u) -> None:
        """Compute y = C*x + D*u and write efforts/flows.

        y has length n_outputs = 2*n_bonds. The C and D matrices are built with
        the efforts occupying the first half of the rows and the flows the
        second half, so the split point (stride) is exactly ``n_outputs // 2``.
        """
        Cx = self._mat_vec(C, x) if x else [0.0] * self.n_outputs
        Du = self._mat_vec(D, u) if u else [0.0] * self.n_outputs
        y = [Cx[i] + Du[i] for i in range(self.n_outputs)]

        stride = self.n_outputs // 2
        for i in range(stride):
            setattr(self, f"e_{i + 1}", y[i])
            setattr(self, f"f_{i + 1}", y[i + stride])

    def get_continuous_state_derivatives(self) -> List[float]:
        A, B, C, D = self._current_matrices()
        x, u = self._state_and_input_vectors()

        Ax = self._mat_vec(A, x) if x else [0.0] * self.n_states
        Bu = self._mat_vec(B, u) if u else [0.0] * self.n_states
        dx = [Ax[i] + Bu[i] for i in range(self.n_states)]

        for i, state_var in enumerate(self.bg.state_vars):
            setattr(self, f"der_{state_var.name}", dx[i])

        # Keep effort/flow outputs consistent with the current state and inputs.
        self._update_outputs(C, D, x, u)

        return dx


if __name__ == "__main__":
    BondGraphSlave(instance_name="ASDF")
