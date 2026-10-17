"""Circuit adapters: op-amp, LDO and bandgap voltage reference."""

from .bandgap import Bandgap
from .base import Circuit
from .ldo import LDO
from .opamp import OpAmp

CIRCUITS = {"opamp": OpAmp, "ldo": LDO, "bandgap": Bandgap}


def get_circuit(name: str) -> Circuit:
    """Return the adapter for 'opamp', 'ldo' or 'bandgap'."""
    try:
        return CIRCUITS[name]()
    except KeyError:
        raise ValueError(f"Unknown circuit '{name}'; choose from {sorted(CIRCUITS)}.") from None


__all__ = ["CIRCUITS", "Bandgap", "Circuit", "LDO", "OpAmp", "get_circuit"]
