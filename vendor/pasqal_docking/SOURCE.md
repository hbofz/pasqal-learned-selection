# Source

- Repository: https://github.com/pasqal-io/Molecular-Docking
- Commit: fafb1f2a872cee0f241cfd233346bc53e39cfcca
- Files: `src/solver/greedy_lattice_mapping.py`, `src/solver/greedy_subgraph_vv.py`, `src/solver/classic_MIS_solvers.py`, and the repository `LICENSE`.
- Unchanged. Imported only through `learned_selection/vendored.py`.
- This is the public code for arXiv:2508.18147, the molecular docking follow-up that reuses the divide-and-conquer heuristic of arXiv:2508.06130.
- The lattice coordinates (`src/solver/coordinate_arrays.pickle`) are not vendored as a pickle. They were decoded once with a restricted unpickler into `learned_selection/data/fresnel_61.json`.
