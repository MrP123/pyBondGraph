import numpy as np
import sympy as sp


def to_numpy(
    M: sp.Matrix,
    subs: dict[sp.Symbol, float],
) -> np.ndarray:
    """Convert a symbolic SymPy matrix to a numeric NumPy array.

    Parameters
    ----------
    M : sp.Matrix
        Symbolic matrix.
    subs : dict[sp.Symbol, float]
        Substitution dictionary mapping symbols to numeric values.

    Returns
    -------
    np.ndarray
        Float64 array.
    """
    return np.array(M.subs(subs), dtype=np.float64)


def to_control_ss(A: sp.Matrix | np.ndarray, B: sp.Matrix | np.ndarray, C: sp.Matrix | np.ndarray, D: sp.Matrix | np.ndarray, subs: dict[sp.Symbol, float] | None = None):
    """Convert state-space matrices to a :class:`control.StateSpace` object.

    Parameters
    ----------
    A, B, C, D : sp.Matrix or np.ndarray
        State-space matrices (symbolic or already numeric).
    subs : dict, optional
        Substitution dictionary.  If provided the symbolic matrices are
        substituted and converted to NumPy arrays first.

    Returns
    -------
    control.StateSpace

    Raises
    ------
    ImportError
        If the ``control`` package is not installed.
    """
    try:
        import control
    except ImportError:
        raise ImportError(
            "python-control is required for to_control_ss(). "
            "Install it with: pip install control"
        )

    if subs is not None:
        A, B, C, D = (to_numpy(M, subs) for M in (A, B, C, D))

    return control.ss(A, B, C, D)
