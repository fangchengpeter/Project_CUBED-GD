# defenses/__init__.py
from .base import BaseDefense
from .no_defense import NoDefense
from .trimmed_mean import TrimmedMean
from .median import Median
from .krum import Krum
from .bridge_b import BridgeB # Make sure Krum, TrimmedMean, Median are also importable for BridgeB
from .geomedian import GeoMedian
__all__ = [
    "BaseDefense",
    "NoDefense",
    "TrimmedMean",
    "Median",
    "Krum",
    "BridgeB",
    "GeoMedian",
]