"""ICA et optimisation stochastique : séparation de sources simulées et PICA sur IRM fonctionnelle."""

from .data import SCENARIOS, make_mixing, make_sources
from .experiments import BenchmarkResult, fit_unmixing, grid_search_sgd, run_benchmark
from .metrics import amari_index
from .sgd_ica import SGDICA, whiten

__all__ = [
    "SCENARIOS",
    "SGDICA",
    "BenchmarkResult",
    "amari_index",
    "fit_unmixing",
    "grid_search_sgd",
    "make_mixing",
    "make_sources",
    "run_benchmark",
    "whiten",
]
__version__ = "1.0.0"
