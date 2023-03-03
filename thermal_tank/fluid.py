from enum import Enum, auto

from scp.ethyl_alcohol import EthylAlcohol
from scp.ethylene_glycol import EthyleneGlycol
from scp.methyl_alcohol import MethylAlcohol
from scp.propylene_glycol import PropyleneGlycol
from scp.water import Water


class FluidType(Enum):
    EthylAlcohol = auto()
    EthyleneGlycol = auto()
    MethylAlcohol = auto()
    PropyleneGlycol = auto()
    Water = auto()


def get_fluid(fluid_type, concentration=None):
    if fluid_type == FluidType.EthylAlcohol:
        return EthylAlcohol(concentration)
    elif fluid_type == FluidType.EthyleneGlycol:
        return EthyleneGlycol(concentration)
    elif fluid_type == FluidType.MethylAlcohol:
        return MethylAlcohol(concentration)
    elif fluid_type == FluidType.PropyleneGlycol:
        return PropyleneGlycol(concentration)
    elif fluid_type == FluidType.Water:
        return Water()
    else:
        raise ValueError("Fluid type not recognized")
