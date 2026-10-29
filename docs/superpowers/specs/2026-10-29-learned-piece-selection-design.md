# Learned piece selection for Pasqal's divide-and-conquer solver: design

Status: draft for Hamzah's review, 2026-10-29. Source plan: `pasqal_research_plan (1).pdf` (October 2026). The older `EXPLANATION.md` (v1, September 2026) is background only, and none of its claims are trusted without re-checking.

## 1. Intent

Pasqal's analog neutral-atom machines solve maximum independent set (MIS) natively. This works only when the conflict graph is a unit-disk graph, meaning two items conflict exactly when their atoms sit within the blockade radius. Most business problems do not have this shape. Cazals et al. (arXiv:2508.06130) proved that converting them with gadgets costs at least quadratically many atoms and can wreck answer quality. They proposed a divide-and-conquer heuristic instead:

1. Carve out a piece of the problem that fits a fixed atom lattice.
2. Solve that piece on the machine.
3. Remove the piece and everything its solution conflicts with.
4. Repeat until nothing is left, then merge the piece solutions.

A small beam search runs over these choices. A fixed greedy rule (GLS, Greedy Lattice Subgraph) decides which piece to take and how to grow it.

**This project replaces that fixed rule with learned models.** It measures whether they give better answers, need fewer quantum runs, or both, and how much model complexity the gain requires.

The work is packaged in an AI agent pipeline: a plain-language request plus a spreadsheet goes in, and a verified answer comes out. It is shown off on portfolio selection, where the task is to find the largest set of stocks whose returns do not move together.

Why learning fits: practice problems can be generated without limit, and the outcome of every choice can be measured exactly.

A research check on 2026-10-29 found no published work that learns this selection step. The 12 papers citing 2508.06130 were checked, and none do. The closest work learns graph partitions for gate-based QAOA (arXiv:2605.13072) or variable elimination in recursive QAOA (arXiv:2207.06294); neither involves an atom lattice.

## 2. Goals

- **G1. Harness.** A test bench where every method runs on identical instances, an identical lattice and an identical piece solver, and is scored the same way. The learned models also train inside it.
- **G2. Baselines.**
  - Pasqal's own GLS code, run as-is.
  - A version faithful to the paper's text.
  - Classical greedy.
  - Strong classical solvers as the reference optimum.
- **G3. A ladder of learned models** (section 6), climbed from cheap to expensive. Each rung has to beat the cheaper ones to justify itself. Hamzah wants as many model types tested as are useful.
- **G4. Answers to the research questions.**
  - Q1. Better answers than GLS?
  - Q2. The same quality with fewer quantum runs?
  - Q3. Which model type is best, and does the complexity pay?
  - Q4. Does a model still work when the problems shift, for example from calm to crisis markets?
- **G5. Portfolio demo** on real S&P 500 prices, calm against crisis periods, including how the chosen portfolios behave afterwards.
- **G6. The agent**, which answers Q5: can it reliably go from a plain request and price data to a verified result run on Pasqal's emulators?

## 3. Non-goals

- Any claim that quantum beats classical, or any claim of quantum advantage.
- Portfolio weighting, meaning how much of each stock to buy. Weighting by expected return (weighted MIS) is a possible extension only.
- Pulse optimisation. Use one standard adiabatic schedule throughout, so that selection is the only thing that varies.
- Free atom placement. Layouts are fixed lattices, and the decision is which traps to fill.
- Real QPU runs. The account has 0 h of QPU credit. QPU validation becomes a goal only if credit arrives.

## 4. Anti-goals: failure even if the code works

- Beating a watered-down baseline. The baseline is Pasqal's own code. Every discrepancy between their code and their paper is documented, and both versions are reported.
- A method that uses exact answers to pieces to decide which piece to send, without counting those solves as quantum runs. See rules R3 and R5 in section 5.
- Methods compared under different lattices, search widths, piece solvers, instance sets or restart budgets.
- Results reported without their settings: graph family, size n, the widths (k, s, l), lattice, restart budget, piece solver, and whether classical time is included.
- A portfolio demo that only works on hand-picked dates or thresholds.
- Mixing up feasibility (does the piece fit the hardware at all) with quality (how good the final answer is).
- Test data leaking into training. This includes market windows that overlap a test window.

## 5. Harness rules and facts the builder must respect

### Lattice and fit

- Pasqal's code (`pasqal-io/Molecular-Docking`, `src/solver/greedy_lattice_mapping.py`) uses:
  - a 61-trap triangular patch with 5.0 µm spacing;
  - a blockade radius of 6.6 µm.
- So only nearest neighbours interact. The next-nearest distance is 8.66 µm.
- A valid piece is therefore a mapping of some vertices onto traps such that two vertices are adjacent in the problem exactly when their traps are lattice neighbours. This is an induced-subgraph embedding, so within a piece no vertex can have more than 6 neighbours.
- Default to Pasqal's 61-trap lattice. Treat larger custom layouts as a swept parameter, and check their limits against the installed Pulser and the Pasqal cloud device specs, not against any document.

### Rules

- **R1. Fit checking.** Every committed piece passes the harness fit checker. A sample of pieces is also built as real Pulser registers and validated against the device.
- **R2. Final answers.** Every final answer is checked to be an independent set of the original graph.
- **R3. Counting quantum runs.**
  - Every call to the piece solver is one quantum run, even when the classical stand-in answers it.
  - Small pieces count too. Pasqal's code brute-forces pieces of 5 nodes or fewer for free. If those stayed free, a method could cut its run count just by aiming for tiny pieces. Also report a second count that follows Pasqal's convention, so our numbers line up with theirs.
  - Q2 counts the runs spent solving one instance. Training and label generation use the classical stand-in, so they cost classical compute, not quantum runs. Report that compute separately as a one-time cost. If a model ever trains against an emulator, report those calls separately too.
  - All classical time is reported separately.
- **R4. Decoding.** Decode Pulser bitstrings by register insertion order (`list(reg.qubits)`), never by sorted qubit names. Sorting diverges from insertion order from 10 atoms upward, and the results then look plausible while being wrong.
- **R5. Two tracks for screening candidate pieces.** Pieces have at most 61 nodes and degree at most 6, so a greedy or local-search independent set of a piece is almost always optimal. Screening with one is effectively a free exact solve.
  - **Main track (structure only).** A method may use piece and graph structure: sizes, degrees, edge counts, how much a piece would remove, and learned predictions. It must not compute an independent set of a candidate piece, or of the remaining graph, outside a counted solver call. Models trained on offline labels are fine, because they predict at run time rather than compute. All headline Q1 and Q2 results come from this track.
  - **Hybrid track.** Classical heuristic screening is allowed, and GLS gets the same screen as its baseline ("GLS + screen"). Results are reported separately and labelled as hybrid.

### Piece solver

- The piece solver sits behind one interface and returns up to s solutions, best first.
- Three implementations:
  1. Exact classical MIS, used for training and for most evaluation. It is fast, because pieces have at most 61 nodes.
  2. Local emulators (Pulser's own emulator, and emu-sv for small pieces).
  3. Pasqal cloud emulators. Emulator hours are unlimited on project `6c44ee19-cbf0-4355-934f-4825f40cca40`. The SDK (pasqal-cloud 0.24.0, login verified 2026-10-29) lists `EMU_MPS`, `EMU_TN`, `EMU_FREE` and `EMU_FRESNEL`. The account can also fetch device specs for `FRESNEL` and `FRESNEL_CAN1`, so the harness can check pieces against real device limits and calibrated layouts even without QPU credit.
- The emulators are used to validate that conclusions drawn with the classical stand-in still hold with quantum-sampled solutions.
- **What "up to s solutions" means.**
  - **Exact stand-in:** the s largest distinct maximal independent sets of the piece, optimum first. Use maximal sets, not only maximum ones, so that branching does not collapse whenever the optimum is unique.
  - **Quantum solvers:**
    1. Repair every sampled bitstring into a maximal independent set: drop conflicting vertices greedily, then fill greedily.
    2. Merge duplicates.
    3. Keep the s most frequent.
    4. Record the invalid-sample rate.
  - **Pasqal's code** returns s diverse greedy maximal sets from its classical generator, and unfiltered top-s bitstrings from its quantum solver.
  - **For comparisons,** every method, GLS included, uses the shared piece solver. The vendored code with its own solvers serves only as a test oracle.
- **Calibrate early.** Right after the baseline runs, solve a sample of pieces with both the stand-in and a cloud emulator, and compare them. Check how often the emulator's best sample is optimal, and how the top-s sets differ. If they disagree materially, add a noisy stand-in fitted to the emulator's statistics, and train against that.

### Known details of Pasqal's code that differ from the paper

These come from a 2026-10-29 code read. Confirm each one by running the code.

- **Ranking.** Candidates are sorted descending by 1 − |avg_lattice_degree − deg(v)|/n, with random tie-breaks. The default path has no "neighbours already assigned" term. The paper says it ranks by degree and by the number of neighbours already assigned.
- **Seed site.** The code seeds at trap 32, which is on the boundary. The paper says the central site.
- **Search loop.** The code's loop and the paper's loop differ in more than one setting.
  - **Code:** `solve_breadthfirst` in `src/solver/greedy_subgraph_vv.py`, the method the run script calls.
    - Every vertex of the remaining graph is tried as a seed, and the k largest pieces are solved.
    - Each branch yields k·s children.
    - Pruning to k happens only when a running count of solved children passes l. That count is not reset between rounds unless pruning fires, so l is a trigger, not a beam width.
  - **Paper:** a random start item, k = 4 pieces, s = 2 answers each, and a beam of l = 10.
  - **Consequences:**
    - At (4, 2, 10) the code settles at about 16 runs per round and the paper at about 40, so equal widths do not mean equal budgets.
    - The seed hook H1 does nothing in the code's loop, because every seed is tried.
- **No restarts as shipped.** The code calls `random.seed(0)` inside every `GreedyMapping`, so it is deterministic and repeat runs give identical results.
- **Small pieces.** Pieces with 5 nodes or fewer are brute-forced.
- **Licence.** The Pasqal Open-Source Software License: MIT-derived, with the patent grant limited to research and academic use. Fine for this project. Keep attribution, and vendor the code unchanged in its own directory.

### Metrics

- Gap to the best known answer, in percent.
- Quantum runs.
- Total atoms used.
- Number of rounds.
- Fit pass rate, which must be 100% for a method to count.
- Classical wall-clock time.

### Reference optimum

- Use an exact solver where it proves optimality within a time limit.
- Otherwise take the best of KaMIS and simulated annealing, labelled "best-known".

## 6. The decomposition engine and where models plug in

The engine is shared by every method. A method differs only in which decision points ("hooks") it replaces.

### State and rounds

- A branch is the pair (remaining graph H, partial solution S).
- Each round, for each branch in the beam:
  1. Generate candidate pieces.
  2. Solve the chosen pieces.
  3. Branch on up to s solutions per piece.
  4. Remove the piece and the conflicts of the chosen solution, matching whatever Pasqal's code does here.
  5. Keep a beam of branches.
- The run ends when every branch's remaining graph is empty.

### Loop modes

- **Code-faithful.** Reproduces `solve_breadthfirst`. With the GLS hooks plugged in and seed 0, it must exactly reproduce the vendored code's output on a fixed instance set. That test is the guard against a watered-down baseline.
- **Paper-faithful.** Follows the paper's text.
- Learned models run on the code-faithful loop. The paper-faithful loop is run for GLS and for the best models only.
- Widths mean different budgets in the two loops, so methods are compared on curves of gap against quantum runs, not at matched widths.

### Restarts

- A restart is a full rerun with a different random tie-break seed. The vendored code fixes the seed at 0, so the restart seed is a parameter of our reimplementation and the vendored copy stays unchanged.
- GLS's curve sweeps both the widths and the number of restarts, with each restart's quantum runs added to the total.
- The teacher model (7) uses the same definition of a restart.

### Hooks

- **H1. Seed:** which vertex starts a piece.
- **H2. Growth order:** which vertex to try adding next.
- **H3. Trap choice:** which lattice site a vertex goes to.
- **H4. Candidate choice:** which generated pieces get solved.
- **H5. Beam pruning:** which branches survive.
- **H6. Search widths:** k, s and l, which may also be chosen per instance or per round.

### Model ladder

All of these were approved by Hamzah on 2026-10-29, cheapest first.

| # | Model | Hooks |
|---|---|---|
| 1 | Linear scorer on hand features | H2, H4 |
| 2 | Gradient-boosted trees, learning to rank (LightGBM) | H2, H4 |
| 3 | Evolved rule: tune GLS's own score weights with CMA-ES (tests whether the fixed rule is just mistuned) | H1, H2 |
| 4 | GNN scorers (GIN, GAT, GraphSAGE): imitate GLS first, then train on choices that led to better final answers | H1, H2, H4 |
| 5 | Value-to-go model: predicts how large an independent set the remaining graph will still yield | H4, H5 |
| 6 | Tree search with learned policy and value (MCTS, AlphaZero style) | H4, H5 |
| 7 | Slow teacher, fast student: an expensive search (deep beam, many restarts) labels decisions offline, and a fast model imitates them (DAgger or expert iteration) | any |
| 8 | Learned piece builder: assigns vertices to traps one at a time (attention or pointer style), trained with RL | H1, H2, H3 |
| 9 | GFlowNet that samples diverse good pieces for the search (stretch goal) | H1, H2, H3 |
| 10 | Policy gradient (REINFORCE, PPO) over piece selection, with reward = final set size minus a cost per quantum run | H4, or H1 to H3 |
| 11 | Contextual bandit that picks a GLS configuration each round | H6 |

Models 5 and 7 can wrap around the others.

### Evaluation shape

- Plot gap against quantum runs as a trade-off curve per instance family. Avoid single-number tables.
- Test generalisation: train on some families or periods and test on held-out ones.
- Use multiple seeds and report confidence intervals.

## 7. Instances and training data

Training data is whatever trains best: Hamzah's call. The evaluation spans several families. The portfolio demo is a showcase, not the only scoreboard.

### Families

- **Erdős–Rényi** at p = 0.25 and p = 0.75, n up to 500. This is Pasqal's setting, so the baseline can be sanity-checked against the paper's qualitative results.
- **Real market graphs** from S&P 500 daily returns over rolling windows spanning many years.
  - Two stocks conflict when their correlation exceeds a threshold θ.
  - θ is swept, and it controls density.
  - Conflicts use positive correlation only: holding negatively correlated stocks together is diversification, not a conflict. The trapped-ion portfolio paper (arXiv:2607.01037) uses |corr| instead, so the harness supports both.
- **Synthetic market-like graphs** from a simulated factor model: a market factor, sector factors and noise give correlations, and thresholding them gives a graph. This supplies unlimited training graphs with hubs and sector clusters, and "crisis" variants with a stronger market factor. Suggested by me, not decided.
- **Structural variety:** power-law and stochastic-block-model graphs.
- **Unit-disk graphs**, the easy geometric case, as a sanity family.

Keep held-out families and held-out time periods for testing.

### Facts from the 2026-10-29 research

- No published density figures exist for roughly 500 US stocks, so they get measured here.
- Crisis windows are much denser at a fixed θ. Correlation jumps after September 2008 and in March 2020 (arXiv:1803.06223, 2308.14830).
- For comparison, Hidaka et al. (arXiv:2308.04769) report about 75% density and an average MIS of 123 on about 2,000 Japanese stocks at θ = 0.23.

## 8. Portfolio demo

### Data

- Daily adjusted closes for S&P 500 constituents from yfinance, cached locally.
- Using today's constituents causes survivorship bias. Either use historical constituent lists or state the bias explicitly.

### Periods

- Train on calm periods.
- Test on crises: 2008–09, March 2020, and the builder may add 2022.

### Evaluation

The chosen stock set is held equal-weighted, and its realised volatility over the following months is compared against:

- the equal-weighted index;
- random sets of the same size;
- the classical best answer;
- GLS's answer.

## 9. The agent (built last)

- **Model:** Claude through the Anthropic API with tool use. This is my call, and Hamzah can override it. It needs an API key, stored in the Keychain or an environment variable.
- **Tools:**
  - load a CSV;
  - build the conflict graph from a plain-language request;
  - check whether the graph fits the machine directly;
  - run the decomposition with a chosen model;
  - run pieces on Pasqal cloud emulators;
  - verify the answer;
  - report it in the user's terms.
- **Prior art to cite and distinguish:** Pasqal's agentic workflow paper, arXiv:2607.25834. It executes known protocols and found that expert review remained essential. Ours formulates business problems from the user's own data and includes the learned splitting step.
- **Q5 evaluation:** a set of plain-language requests with CSVs, scored on success rate, with automatic verification of every answer.

## 10. Order of work

1. Harness, kept minimal: the engine in code-faithful mode, the shared piece solver, the fit checker, the metrics, and the exact-match test against the vendored code. Reproduce GLS's qualitative behaviour on Erdős–Rényi graphs.
2. The paper-faithful loop, classical greedy, and the reference solvers.
3. Calibrate the classical stand-in against a cloud emulator on a sample of pieces (section 5).
4. Models 1 to 3, then 4 and 5, then 7, then 6, 8, 10 and 11, then 9. The order within the ladder is flexible. The rule is that a model's results are only trusted once the cheaper models have been measured on the same footing.
5. Validate full runs of the best models on Pasqal cloud emulators.
6. Portfolio demo.
7. The agent.
8. Paper.

Hamzah wants to focus on the learned models first. The harness comes first only because the models need it to train and be scored. Keep it to what models 1 to 3 need, and grow it when a later model needs more.

## 11. Constraints

- **Python:** 3.12, managed with uv. The Mac's system Python is 3.15, which key libraries may not support.
- **Libraries:** Pulser, pasqal-cloud, networkx, numpy, PyTorch, LightGBM and yfinance. Add others only when a rung needs them.
- **Compute:**
  - Mac M2 with 8 GB RAM and 32 GB free disk: development only.
  - Dell G15 (`ssh dell-linux`, WSL2): 12 cores, 11 GB RAM, GTX 1050 Ti with 4 GB, 926 GB free. Long runs and small GNNs.
  - aspire (`ssh bb@aspire`): 4 cores, about 5 GB of RAM free, no GPU. It has a terminal Colab CLI (`~/projects/clis/colab/colab`) for GPU jobs.
  - Pasqal cloud emulators: unlimited hours.
- **Credentials:** never in code. Pasqal cloud login goes in the macOS Keychain under service `pasqal-cloud`.
- **Prose:** no em dashes in anything written for this project.
- **Git:** v2 becomes its own git repository. It currently sits untracked inside the v1 repository. Nothing is committed until a change is verified and Hamzah confirms.
- **v1 code** (`../*.py`) is not read or copied. v2 is a deliberate fresh start.

## 12. Decided by Hamzah

- Fresh rebuild in v2.
- The model ladder in section 6, and he is open to more.
- Portfolio and crisis tests are a demo. Evaluation spans families, and training data is whatever trains best.
- Timing and deadlines are out of scope for this design.
- The compute and cloud access listed above.

## 13. My calls, flagged for review

- Pasqal's own code as the primary baseline, plus a paper-faithful variant.
- Learned models run on the code-faithful loop. Comparisons are made on curves of gap against quantum runs, not at matched widths.
- Rule R3 for counting quantum runs: small pieces count, and training cost is reported separately from Q2.
- Rule R5: headline results come from the structure-only track, and hybrid screening is a separate, labelled track.
- The meaning of "s solutions": maximal sets, with quantum samples repaired. Calibrating the stand-in against the emulator happens early, not at the end.
- A restart is a rerun with a new tie-break seed.
- Positive-only correlation conflicts by default.
- The synthetic factor-model family.
- Claude as the agent's model.
- The libraries.
- Python 3.12.
- This document's location.

## 14. Left to the builder

- Feature sets, model sizes and hyperparameters.
- Training-set sizes and mixtures, within the held-out rules.
- Instance sizes for training.
- File formats and caching.
- The choice of exact solver (CP-SAT or Gurobi academic).
- The threshold grid.
- Window lengths.
- Exact crisis dates.
- Which machine runs what.
