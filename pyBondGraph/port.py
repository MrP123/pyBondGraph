"""Port definition for exposing connection points on sub-models.

A port marks a boundary junction on a sub-model that can be connected
to another sub-model's port via a bond.
"""

from .core import Junction


class Port:
    """An open connection point on a SubBondGraph.

    A port wraps a boundary junction and gives it a name so that
    sub-models can be connected by referring to port names rather
    than internal junction objects.

    Parameters
    ----------
    name : str
        Human-readable identifier for this port (e.g. "mechanical_out",
        "electrical_in").
    junction : Junction
        The boundary junction inside the sub-model that this port
        exposes.  When two sub-models are connected, a new bond is
        created between their respective port junctions.
    domain : str, optional
        Physical domain label (e.g. "mechanical", "electrical",
        "hydraulic").  Used for optional compatibility checking
        when connecting ports.  Defaults to ``"generic"``.
    """

    def __init__(self, name: str, junction: Junction, domain: str = "generic"):
        if not isinstance(junction, Junction):
            raise TypeError(
                f"Port junction must be a Junction instance, got {type(junction).__name__}"
            )
        self.name = name
        self.junction = junction
        self.domain = domain

    def __repr__(self) -> str:
        return f"Port(name={self.name!r}, junction={self.junction}, domain={self.domain!r})"
