# Execution ledger, plan: docs/superpowers/plans/2026-10-29-harness.md
Spec: docs/superpowers/specs/2026-10-29-learned-piece-selection-design.md
Branch: plan-1-harness (created from main at 0e600dd; plan committed as daca3ad)
Executor: inline (executing-plans), Hamzah chose Native 2026-10-29. Commit locally per task; push only after Hamzah sees results.

## Pre-flight (shared interfaces)
| Producer -> Consumer | Produced | Consumed | Found |
|---|---|---|---|
| T1 -> T2 | load_coords, greedy_subgraph_solver_vv | same (tests) | consistent |
| T1,T2 -> T3 | GreedyMapping, Lattice, greedy_subgraph_solver_vv, weighted_generate_different_mis, load_coords, lattice_graph, is_valid_piece, BLOCKADE_UM; first_maximum_set, is_independent, top_maximal_sets, min_degree_greedy | same | consistent |
| T3,T1 -> T4 | run_beam, RunResult, Generator, Pruner, exact_piece_solver, gls_generator; CENTER_TRAP, BLOCKADE_UM | same | consistent |
| T2-T4,T1 -> T5 | erdos_renyi, run_code_faithful, run_paper_faithful, best_of_restarts, exact_mis, min_degree_greedy; weighted_greedy_independent_set | same | consistent |
| T1-T3 -> T6 | PieceSolver, Mapping, gls_generator, repair, is_independent, top_maximal_sets, load_coords | same | consistent |

## Progress
Task 1: Ruling: vendored from the existing scratchpad clone at the pinned commit instead of a fresh /tmp clone; byte-identical verified with cmp; cost if wrong: none
Task 1: Ruling: added learned_selection/data/__init__.py so importlib.resources can address the JSON; cost if wrong: none
Task 1: complete (commits daca3ad..d7949ad, tests: uv run pytest tests/test_lattice.py -q → 4 passed in 0.88s)
Task 2: Ruling: top_maximal_sets runs CP-SAT single-worker with random_seed=0 (plan said "same model"), so tied optima resolve identically every run and engine results are reproducible; cost if wrong: slower piece solves on large pieces
Task 2: complete (commits d7949ad..2aeaba2, tests: uv run pytest tests/test_solvers.py -q -p no:warnings → 9 passed in 1.01s)
Task 3: Ruling: run_beam takes an optional lattice= (default fresnel_61) for the R1 fit check, since the plan's signature had no way to know the generator's lattice; cost if wrong: one extra argument
Task 3: Ruling: added 6 PRUNING_CASES (n=120-150) to the exact-match test. Mutation check showed the plan's 54 small cases cannot detect wrong pruning (prune-to-l missed 24/24), because on those graphs only round 1 has solver-sized pieces. On the added cases prune-to-l, always-k and no-counter-reset are each caught; cost if wrong: +17 s of test time
Task 3: Finding: vendored breadth-first loop explodes on sparse ER (n=200, p=0.05: 4096 branches by round 7), because small-piece children never count toward pruning. Matters for sparse market graphs (plan 2). Our engine raises instead of hanging.
Task 3: complete (commits 2aeaba2..95a8f8d, tests: uv run pytest -q -p no:warnings → 82 passed in 24.84s)
Task 4: complete (commits 95a8f8d..7dc9343, tests: uv run pytest -q -p no:warnings → 87 passed in 27.65s)
Task 5: Finding: the plan's Expected (gls-code beats greedy-pasqal) does not hold. ER n=100 p=0.25 seeds 0-9, (4,2,10), 1 restart, all CP-SAT proven: gls-code 15.27%, gls-paper 14.75%, greedy-pasqal 13.06%, greedy-mindeg (20 seeds) 4.74%. Not tuned, per plan. Cross-checks: Pasqal's pipeline as shipped (their classical piece solver) 15.41%; gls-code best of 20 restarts 8.24%. Plan reviewer independently got 14.1/14.8/13.1/4.7. Report to Hamzah.
Task 5: Ruling: greedy-mindeg records its real budget (restarts*20 seeds) in the restarts field; budgets are not matched because greedy spends no quantum runs; cost if wrong: a reader compares unequal restart budgets without noticing
Task 5: complete (commits 7dc9343..74da019, tests: uv run pytest -q -p no:warnings → 88 passed in 23.98s)
Plan review (fresh agent, rebuilt Tasks 1-4 independently) findings applied:
Review: Ruling: #1 CP-SAT nondeterminism already fixed in Task 2 (single worker); added test_top_maximal_sets_is_deterministic_on_gls_pieces, RED with 8 workers, GREEN with 1; cost if wrong: none
Review: Ruling: #4 added test_code_prune_running_counter (catches a per-level counter) and test_only_solver_children_count_toward_pruning (catches small-piece children being counted); complements Task 3 PRUNING_CASES; cost if wrong: none
Review: Ruling: #5 paper-faithful mode keeps the vendored ranking score. arXiv:2508.06130v2 text, fetched 2026-10-29: "prioritized using a ranking (such as vertex degree and number of neighbors already assigned)" and Algorithm 1 "Sort U by ranking strategy (e.g. degree)"; both are examples, no key or direction pinned. Plan's citation of spec section 5 for this was wrong, the conclusion stands. Documented as a code-vs-paper gap for the paper; cost if wrong: paper-faithful ranking differs from what the authors ran, which nobody can know without their paper-era code
Review: Ruling: #7 max_branches default lowered 10_000 -> 1_000; legitimate beams stay far below (paper mode <= l*k*s), and 10k branches would take hours before raising; pytest-timeout not added (guard test runs in <1 s); cost if wrong: a legitimate huge beam raises and needs the argument raised
Review: Ruling: #8 R1/R2 already explicit raises; run_beam already takes lattice=; added _check_trap_order guard (ValueError) in both generators, test_generators_require_trap_ids_in_order RED->GREEN; cost if wrong: none
Review: Ruling: #10 strengthened test_own_rng_survives_vendored_reseeding (global reseeds around calls, covers paper_generator); cost if wrong: none
Review: Ruling: #2, #9 apply in Task 6 (calibration metrics vs a random-bitstring control; trap_map/layout API). #3 recorded as a Task 5 finding. #6 recorded as a Task 5 ruling.
Review fixes: suite 92/92 green
Task 6: Finding: FRESNEL's only calibrated layout is TriangularLatticeLayout(120, 5.0um). It contains all 61 fresnel_61 traps at identical coordinates (no rotation), but min_layout_filling 0.35 means registers on it need 42-60 atoms, so typical GLS pieces (6-20 atoms) cannot run on the real QPU as-is. Its limits are also lower than Pasqal's schedule: max_amp 11.31 (schedule peaks at 12.57), max_abs_detuning 62.83 (schedule starts at -125.66). Pinned by test_fresnel_rejects_pieces_under_42_atoms. Report to Hamzah.
Task 6: Ruling: emulator sequences are built on Pulser's AnalogDevice instead of FRESNEL. Its TriangularLatticeLayout(61, 5um) has exactly the fresnel_61 ids and coordinates, it has no minimum fill, and Pasqal's schedule fits its limits unclipped; max fill 0.5 caps emulated pieces at 30 atoms (larger pieces are skipped in calibration and counted). The plan said stop if no layout matched; a layout does match (FRESNEL, as a subset), and the blocker is the fill rule, which has a clear path, so this is a ruling, not a stop. Cost if wrong: emulated physics is AnalogDevice's, not FRESNEL's
Task 6: Ruling: the calibration decision rests on pre-repair stats plus a uniform-random-bitstring control (plan review #2): stand-in OK iff emulator top-1 repaired set is optimal on >= 90% of pieces AND emulator raw optimal-shot rate beats random on >= 90%. summarize_counts is a pure function shared by the solver and the control. Cost if wrong: a threshold choice that Hamzah may want to set differently
Task 6: Ruling: trap_map(device) returns a dict; calibrated_layout(device) returns the matching layout (plan review #9); the device spec JSON is written as the raw string; cost if wrong: none
Task 6: smoke test: 3-atom path on EMU_MPS, 500/500 shots optimal {0,2}, 65.7 s round trip, batch 2ad1248f-406e-4468-9d43-2dc2656e363d
Task 6: calibration result: 40 pieces (11-20 atoms), EMU_MPS on AnalogDevice, 500 runs: emulator top-1 optimal 100%, raw optimal-shot rate 0.987 (random control 0.001), invalid 0.000; verdict stand-in OK. Random bitstrings plus repair reach top-1 optimal on 90%, so repair does most of the work at these sizes. Report to Hamzah.
Task 6: complete (commits eb4d393..621b20b, tests: uv run pytest -q -p no:warnings → 98 passed in 30.47s)
Final review (fresh opus reviewer, daca3ad..621b20b): verdict "numbers correct and reproducible; fix C1, I1, I2 before plan 2; scope I3".
Final: fixed C1 string-label views irreproducible across processes. run_beam and the order-dependent solvers now use solvers.canonical (repr-sorted nodes and edges) when any label is not an int. test_string_label_views_reproduce_across_processes (3 PYTHONHASHSEEDs, engine + top_maximal_sets) RED->GREEN. test_engine_on_string_labels now uses zero-padded labels so repr order equals int order.
Final: fixed I1 code-faithful mismatched vendored on int graphs with gaps or shifted labels (31/36). Relabel only when some label is not an int. test_matches_vendored_on_gapped_and_shifted_int_labels (36 cases) RED (31 fail) -> GREEN.
Final: fixed I2 engine trusted generator and solver output. Raises ValueError on an empty mapping, no sets, an empty set, a set outside the piece, or a dependent set. test_engine_rejects_bad_solver_output[none,outside,dependent] RED->GREEN; [empty] and test_engine_rejects_empty_generator_output hang on the old code (shown by the reviewer, queue stable at 16), GREEN now.
Final: suite 140/140 after C1/I1/I2.
Final: minor (deferred): M1 atoms counts only solver-sized pieces; add atoms_all. Leftover solves are counted as quantum runs but never fit-checked (spec R1 vs R3 tension)
Final: minor (deferred): M2 max_branches not exposed by run_code_faithful / run_paper_faithful
Final: minor (deferred): M3 bench records lack small, blockade, max_branches, CP-SAT time limit, git commit
Final: minor (deferred): M4 code_prune counter and paper_generator rng carry state across runs; global RNG makes threaded engine runs nondeterministic; document or construct inside run_*
Final: minor (deferred): M5 bare assert in top_maximal_sets is stripped under -O
Final: minor (deferred): M6 spec section 5 says Pasqal's shipped code is deterministic; false for their run script, which loads graphml (string labels): vendored loop on string-labelled ER(60,0.25) gave 13,13,13,12 across hash seeds. Note in the paper.
Final: minor (deferred): M7 test_engine_accepts_short_solver_output checks only independence, not branch count
Final: Ruling: reviewer "declined to judge" items. Code-faithful cost (10 s per generate call at n=300, p=0.25) is vendored behaviour; a faster verified port is a plan-2 task if training needs it. The engine ignores weights, because the spec is unweighted. The paper loop draws k random starts, a reasonable reading of "a random start item" per candidate. Cloud failure handling (a None result raises) is acceptable for a calibration script. The sparse-graph explosion is already a plan-2 decision. Cost if wrong: plan 2 inherits a slow baseline generator
Final: fixed I3 calibration verdict over-claimed. The verdict is now scoped to the tested atom range; top-1 Jaccard (exact vs emulator) is recorded; a medium set (ER n=300, p=0.25, seeds 0-9) was added. Result on 40 pieces of 25-27 atoms: emulator top-1 optimal 100%, raw optimal-shot rate 0.860 (random 0.000), random plus repair top-1 optimal 68%, invalid 0.002, top-1 Jaccard 0.717 (equally optimal but different sets). Verdict: stand-in OK for 25-27 atoms. Pieces over 30 atoms (the AnalogDevice cap) remain unvalidated (plan 2). Not a RED->GREEN code test: this is a measurement run; the scoped verdict line is the check.
Final: Ruling: no test written for jaccard (one line) or the scoped verdict string; ponytail rule that trivial one-liners need no test; cost if wrong: a typo in a report line
Final: suite 140/140; cloud jobs sent this plan: 81 (1 smoke test, 40 small, 40 medium)
