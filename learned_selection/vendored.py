"""Pasqal's GLS code, imported unchanged from vendor/pasqal_docking (see SOURCE.md there)."""
import os, sys, types
from pathlib import Path

# Notebook runtimes set an inline plotting backend that this project's environment lacks.
if os.environ.get("MPLBACKEND", "").startswith("module://matplotlib_inline"):
    os.environ["MPLBACKEND"] = "Agg"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "pasqal_docking"))
# classic_MIS_solvers imports cplex at top level; we never call its cplex solver.
sys.modules.setdefault("cplex", types.ModuleType("cplex"))

from greedy_lattice_mapping import GreedyMapping, Lattice  # noqa: E402
from greedy_subgraph_vv import greedy_subgraph_solver_vv  # noqa: E402
from classic_MIS_solvers import weighted_generate_different_mis, weighted_greedy_independent_set  # noqa: E402

__all__ = ["GreedyMapping", "Lattice", "greedy_subgraph_solver_vv",
           "weighted_generate_different_mis", "weighted_greedy_independent_set"]
