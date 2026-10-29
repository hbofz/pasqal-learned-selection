# Learned subgraph selection for neutral-atom quantum optimization

**Canonical project document.** This is the single source of truth for what this project is, what has been decided, why, and what is still open. Humans and agents both read it. If something here conflicts with a conversation, a memory, or an earlier file, this document wins.

- **Owner:** Hamzah, St. Olaf College. Senior, CS and quantitative economics.
- **Supervisor:** Dr. Michael Haydock (IBM Fellow Emeritus). Academic advisor: Prof. Ryota Matsuura, MSCS Department Chair.
- **Course:** Independent study, Fall 2026. Graded (letter grade). Form signed and submitted 2026-09-04.
- **Deliverable:** A research paper of peer-reviewed journal quality, plus a demonstration on two contrasting problem instances.
- **Last substantive update:** 2026-09-06, after the first end-to-end Pulser run (D12, D13).

### How to maintain this document

Add to the decision log rather than silently editing decisions. When a decision is reversed, mark the old entry superseded and write a new one with the date and reason. Section 8 tracks which factual claims have been verified against primary sources and which have not; move claims between those lists as they are checked. Do not delete rejected directions from section 6, since the reasons they were rejected are what stop them being re-proposed.

---

## 1. What this project is, in one paragraph

Neutral-atom quantum computers solve independent-set problems by physically arranging atoms so that the geometry encodes the problem's constraint graph. Producing a valid arrangement is NP-hard and acts as a feasibility gate: a graph that cannot be embedded cannot run at all. Rather than attacking the placement problem directly, which is now contested, this project targets the **selection** decision inside Pasqal's published divide-and-conquer embedding scheme, where a greedy rule currently chooses which subgraph to extract next. The project builds a reproducible benchmark harness for embedding strategies, then replaces that greedy rule with a learned policy and measures the difference. The contribution is to the compilation layer. No claim is made about quantum speedup.

---

## 2. Decision log

| # | Date | Decision | Rationale |
|---|---|---|---|
| D1 | 2026-09-04 | Independent study accepted; topic is quantum computing plus agentic AI applied to an optimization problem | Supervisor's proposal; form signed |
| D2 | 2026-09-04 | Target problem left open by supervisor; transportation optimization is not binding | Supervisor's email: "feel free to change out of the transportation optimization into something that really peaks your interest" |
| D3 | 2026-09-05 | Reject the classical transportation problem as the target | It is a linear program solvable in polynomial time by network simplex. No hardness, therefore no defensible quantum framing |
| D4 | 2026-09-05 | Relocate the novel contribution from the agentic formulation layer to the embedding layer | LLM-to-QUBO autoformulation is published prior art (arXiv:2509.00099 and the OptiMUS / ORLM / LLMOPT / NL4Opt lineage). Those systems stop at the QUBO matrix because they target annealers. Neutral atoms require an additional geometric step that none of them handle |
| D5 | 2026-09-05 | Run a full literature review before writing code | Correct call. It surfaced D6 and D7, either of which would have wasted six weeks |
| D6 | 2026-09-05 | **Reject "learn a better atom placement" as the primary contribution** | The slot is contested. DEN/ELF (arXiv:2605.03565, 2605.03503) occupies neural placement. GAGE inside the Quantum Compilation Toolkit (arXiv:2412.14976, Phys. Rev. Research 7, 033107) already learns optimized atom positions via random-key optimization with genetic and annealing updates against a graph-edit-distance cost. Pasqal's own team publishes in this area |
| D7 | 2026-09-05 | **Reject spectral-gap-optimized embedding as the primary contribution** | Evaluating the minimum gap requires many-body spectra that scale exponentially, capping instances at roughly 20 atoms. Any claim would concern toy systems. Also partially occupied by Bombieri et al. (arXiv:2411.04645, PRX Quantum 6, 020306). Retained as an optional stretch extension only |
| D8 | 2026-09-05 | **Primary contribution: a learned policy for the subgraph-extraction decision inside divide-and-conquer embedding** | The slot is empty. No published work applies RL or a learned policy to analog Rydberg register construction or to subgraph selection. Pasqal published their selection rule as an explicitly greedy baseline, which signals they have not closed the question internally. The learning problem is classical graph combinatorics, matching the owner's skills |
| D9 | 2026-09-05 | **Sequence the work benchmark-first, learning-second** | The benchmark harness is publishable on its own. If the learned policy underperforms, the semester still produces a result. This is the governing risk decision for the project |
| D10 | 2026-09-05 | Begin the learned policy by imitating the greedy baseline, then fine-tune | Higher probability of working and far easier to debug than training a policy from scratch |
| D11 | 2026-09-05 | Treat QPU access as validation only, never as a dependency | Access is being requested through the supervisor's contact with Pasqal's CEO and is not guaranteed. Mirrors how the baseline paper validated: 10 instances on hardware against 100 in simulation |
| D12 | 2026-09-06 | **SUPERSEDED by D14, same day.** ~~Design for a 30-atom piece ceiling, not 80~~. The 30 figure was the shipped convenience layout, not the hardware | Verified against Pulser 1.9.0. `AnalogDevice` admits only registers carved from its one calibrated layout, `TriangularLatticeLayout(61, 5.0µm)`, and `max_layout_filling` of 0.5 binds before `max_atom_num` of 80: 31 atoms is rejected. Consequence: piece size is small and fixed, so the number of divide-and-conquer rounds, not the size of any one piece, is what selection actually controls. An n=500 instance needs at least ⌈500/30⌉ = 17 pieces even in the impossible best case, and realistically far more. This makes QPU or emulator call count (section 6) the primary metric rather than a secondary one, and it raises the value of the "match GLS at a lower call count" win condition relative to the optimality-gap one |
| D13 | 2026-09-06 | **Phase 2 trains against a classical MWIS sub-solver; emulation is validation only** | Measured QuTiP state-vector cost on the owner's machine: 2.2 s at 12 atoms, 8.4 s at 14, 41.7 s at 16, with 18 failing to complete in 11 minutes (section 7). A policy-learning loop needs thousands of rollouts and cannot sit on that. A classical oracle makes the reward exact and fast, removes sampling noise from the training signal, and reframes the contribution as a classical decision policy for quantum embedding, which is still novel and still the empty slot from D8. Emulator and QPU runs then validate the selected pieces rather than score them. Closes open question 4 |
| D14 | 2026-09-06 | **Piece ceiling is 80 atoms on a custom layout, superseding D12** | D12 read `max_layout_filling` of 0.5 against the 61-trap layout Pulser ships and concluded 30. That layout is a convenience default, not the machine. Pasqal's Orion Beta technical overview states layouts on demand, minimum 10 and maximum 200 traps, and 100 qubits. Verified directly: `AnalogDevice` accepts a custom `TriangularLatticeLayout(200, 5.0)` and holds 80 atoms on it, where `max_atom_num` binds before the filling fraction. 250 traps is rejected on the 38 µm radial limit. Consequence: the lattice is a parameter of the harness, not a fixed given, and GLS must be reproduced with it configurable. D12's downstream claim that call count matters more than piece size is weakened but not void, since D15 still forces decomposition |
| D15 | 2026-09-06 | **Every benchmark instance requires decomposition, on degree rather than capacity** | A triangular lattice has at most six neighbours within the blockade radius at any lattice size, verified at 61 and 200 traps. A vertex of degree 7 or more therefore has no faithful placement anywhere on it. All eleven instances in the library carry such vertices, including the 29-node `king-s6-f80` that fits the capacity ceiling with room to spare. This is a necessary condition checkable without search, and it is the structural reason divide-and-conquer is the only route on this hardware, independent of how large the layout gets |
| D16 | 2026-09-06 | **Refines D8. The contribution is not a better ranking rule, it is that good ranking makes the candidate budget pay** | D8 assumed the greedy selection rule was leaving performance on the table. Partly true, but measurement reframes it. Ranking accuracy on held-out decisions is 78.1% for the baseline rule, 95.3% for seven linear hand features and 96.9% for a GNN, so selection is nearly solved by trivial means. The gap persists because a scorer can only choose among the candidates it is shown. Raising k, the candidates generated per round, from 4 to 16 leaves the baseline flat (8.7% to 7.9%) and takes the learned policy from 6.0% to 2.9%, crossing the classical randomised greedy floor of 4.8% at k=8 and holding on a shifted set. QPU calls rise only 3.7 to 4.3, because candidates are generated and screened classically and only the committed piece is dispatched. **State the claim as: a learned selector converts classical candidate generation into solution quality at near-constant quantum cost.** That is stronger and more honest than "beats the greedy rule by 27%" |
| D17 | 2026-09-06 | **Stop escalating model capacity. Hand features are sufficient** | A GNN over the graph each decision acted on, trained listwise on Monte Carlo sibling values, reaches 96.9% ranking accuracy against the linear model's 95.3% and is indistinguishable end to end (2.8% against 2.9% at k=16, 6.0% against 6.2% shifted). REINFORCE on top of it moved mean episode return from 12.06 to 12.06. Both were built and measured, not assumed. Torch is retained as a dependency only because `gnn.py` documents this negative result; nothing in the main path needs it. Spend remaining effort on candidate generation (D16), not on model capacity. Closes open question 3 |
| D18 | 2026-09-06 | **Retune k, s and l. The paper's values were fitted to a rule that cannot exploit a wider search** | arXiv:2508.06130 fixes k=4, s=2, l=10. Measured here, the baseline greedy rule is flat or worse as k rises (8.7% to 7.9% from k=4 to 16), so those widths are the right choice FOR THAT RULE. With the learned selector every width pays. Sets per mapping is the strongest single lever (s=4 against s=2 takes 2.8% to 2.1% at k=16) and beam width is effectively free, since raising l slightly REDUCES QPU calls as better branches finish sooner. At k=32, s=4, l=20 the held-out gap is 1.0%, verified over three disjoint seed blocks with ranges non-overlapping between tiers. On the benchmark library this gives 1.3% on the abstract family against the GLS baseline's 6.7% and the classical greedy floor's 4.8%, and 0.0% on the geometric family, which GLS did not reach. **Report the widths with every result**, since a comparison at the paper's widths and one at these widths are different experiments |
| D19 | 2026-09-06 | **Frame the geometric demonstration as evidence about feasibility, never about advantage** | The EV charging siting instance (`ev.py`, real OpenStreetMap coordinates for three metros) is a unit-disk graph by construction, so it is the best case this hardware has. Measured across redundancy radii from 300 m to 8 km on all three cities, classical randomised greedy reaches the exact optimum or within a fraction of a vertex every time. Unit-disk MIS admits a PTAS, so this is theory appearing in measurement rather than a surprise. At the reported 5 km radius the learned selector cuts the baseline's gap from 5.4% to 1.6% and still loses to classical greedy at 0.6%. **The instances that embed well on Rydberg hardware are precisely the ones classical algorithms already solve, and the classically hard instances are precisely the ones that will not embed.** That tension is the honest headline of the geometric demonstration and it reinforces section 9 rather than threatening it. The method contribution has to rest on the abstract family |

### Decisions still open

See section 7. The problem pair, the paper framing and venue, and the exact policy architecture are not yet decided.

---

## 3. Ground truth: the technical situation

### 3.1 Hardware model

Pasqal builds neutral-atom quantum processors. Atoms are trapped in 2D by optical tweezers. Ground state encodes 0, Rydberg state encodes 1.

The **Rydberg blockade** is the load-bearing mechanism: two atoms within a fixed distance cannot both occupy the Rydberg state. This is physics, not programming. It is structurally identical to the independence constraint in a graph, so Maximum Independent Set is encoded natively by atom placement, with no penalty terms.

The register geometry is reconfigurable shot to shot. This is Pasqal's distinguishing capability against fixed-topology hardware, and it is simultaneously the origin of the embedding problem. Advantage and bottleneck are the same fact.

### 3.2 The embedding problem

A graph runs in analog mode only if it can be represented as a **unit-disk graph**: vertices placed in 2D such that an edge exists **if and only if** two vertices are within the blockade radius.

Two silent failure modes:
- **Phantom edge:** unrelated atoms land within the radius. A constraint appears that was not in the problem.
- **Missing edge:** conflicting atoms are placed too far apart. A real constraint vanishes.

Neither errors. The machine simply solves a different problem.

Established facts:
- Finding UDG representations for generic graphs is NP-hard.
- Not all graphs admit a UDG representation. Counterexample: a star K(1,n) for large n, where the center must be near all leaves while the leaves stay far from each other.
- Therefore embedding is a **feasibility gate**, not a performance knob.

### 3.3 The hardness result that shapes everything

Cazals et al. (arXiv:2508.06130, Phys. Rev. A 112, 062416, published 2025-12-08) prove that any polynomial reduction from arbitrary graphs to unit-disk instances incurs a quadratic blow-up in vertex count and degrades approximation guarantees.

Practical consequence, from arXiv:2412.14976: faithful embedding of n logical variables needs roughly N ≲ 4n(n−1) physical qubits. Ten logical variables needs around 360 physical qubits, which exceeds current devices at roughly 250.

Additional consequence noted in arXiv:2609.01248: gadget mappings can close the spectral gap even when the original instance is classically easy. A bad embedding does not only waste atoms, it can make an easy problem take exponentially longer.

### 3.4 The baseline this project targets

The same Cazals et al. paper proposes the alternative: a divide-and-conquer heuristic with only linear overhead that iteratively extracts unit-disk-embeddable subgraphs, solves each on precalibrated atomic layouts, and merges the independent sets into a global solution.

Note "precalibrated atomic layouts." They do not place atoms freely at all. They use fixed lattices and decide which pieces of the problem to map onto them.

The selection rule is called **Greedy Lattice Subgraph (GLS)**. It seeds at the central lattice site, grows outward by exploring unmapped neighbors of already-placed nodes, and ranks candidates by a fixed rule using vertex degree and the number of neighbors already assigned. A mapping is accepted only if geometric adjacency and non-adjacency are both preserved. Reported hyperparameters: k=4, s=2, ℓ=10 for breadth-first branch retention. The triangular lattice produced the largest subgraphs roughly 60% of the time.

Benchmarks: Erdős-Rényi graphs at densities p=0.25 and p=0.75, up to n=500, averaged over 100 instances in simulation and 10 on the Orion Alpha processor. Reported to achieve a smaller optimality gap against CPLEX and simulated annealing than a classical greedy baseline.

**GLS is deterministic, hand-tuned, and explicitly greedy. It is the thing this project replaces.**

The same heuristic is reused unchanged in the molecular docking follow-up (arXiv:2508.18147) and the iterative optimization paper (arXiv:2603.28933), which means improving it has reach beyond one paper.

---

## 4. Prior art landscape and slot occupancy

| Approach | Representative work | Overhead | Occupancy |
|---|---|---|---|
| Force-directed layout (Fruchterman-Reingold) | arXiv:2602.15487 via NetworkX | O(n) atoms, no feasibility guarantee | De-facto default. Weak baseline |
| Nonlinear-solver placement | Ipopt-style, referenced in DEN work | O(n) | Occupied, and possibly harder than the original problem |
| Neural / gradient placement | DEN + ELF, arXiv:2605.03565, 2605.03503 | O(n) | **Occupied.** 76% of PROTEINS graphs on Aquila; all Turin antenna subgraphs on Orion Alpha |
| Learned placement via metaheuristic | GAGE in arXiv:2412.14976 | O(n) | **Occupied.** Random-key optimization with genetic and annealing updates |
| Gadget-based exact reduction | Nguyen et al., arXiv:2209.03965, PRX Quantum 4, 010316 | O(n²), proven | Occupied and provably expensive |
| Rydberg quantum wires | arXiv:2109.03517 | Ancilla chains | Occupied |
| Parity / LHZ architecture | arXiv:2210.05604, PRL 130, 220601; anchor atoms arXiv:2410.03902 | Constant per constraint | Occupied |
| 3D register embedding | arXiv:2209.05164 (degree ≤6); newer scheme at O(min{m√n, n²}) | Sublinear ancilla | Occupied |
| Divide-and-conquer decomposition | **arXiv:2508.06130 (GLS)** | O(n) | Method occupied. **Selection rule is greedy and unoptimized** |
| Hybrid decomposition (Benders, cutting planes) | arXiv:2402.05748, Phys. Rev. A 110, 012434; enhanced arXiv:2503.03518; LP-Quts arXiv:2603.28933 | Varies | Occupied |
| Gap-aware geometry | Bombieri et al., arXiv:2411.04645, PRX Quantum 6, 020306 | n/a | Partially occupied, by hand-designed geometry changes |
| **Learned selection / RL for analog embedding** | **none found** | n/a | **OPEN** |

### 4.1 The near-collisions, and why they do not block this

**arXiv:2506.05536 (QC-Daemon, "Atom Game").** An RL agent with two transformer variants for *digital circuit compilation* on reconfigurable atom arrays. It solves move synthesis: orchestrating parallel gate circuits by physically rearranging atoms to reduce logarithmic infidelity, benchmarked to 100 qubits. It is not analog register embedding and not MIS/MWIS register construction. Cite it, do not fear it.

**arXiv:2607.25834 (Pasqal agentic workflow, Dalyac, Dauphin, Henriet, submitted 2026-07-28).** An LLM agent carries a task from prompt to QPU execution on the FC1 (Canada) and SA1 (Saudi Arabia) machines, validated on two many-body simulations and a graph-coloring optimization. A second agent classified 633 Rydberg-array papers; of 526 classified, roughly half were judged viable starting points on present hardware. The paper is candid that human intervention was crucial for scientific validity: the agent selected an inadequate observable in one experiment and produced a plausible but incorrect hardware diagnosis in another, both caught only by expert review. **It executes known protocols and does not select embedding strategies.** Raise this with the supervisor directly rather than letting him discover it.

### 4.2 Transferable methodology from adjacent domains

These are the templates to borrow, since the neutral-atom slot is empty:
- **D-Wave minor embedding with RL:** CHARME (arXiv:2406.07124, GNN plus RL ordering policy) and Nembrini et al. (arXiv:2507.16004, PPO, Quantum Machine Intelligence, DOI 10.1007/s42484-026-00341-4). Closest problem shape, though the target hardware graph is fixed and sparse rather than continuous 2D.
- **VLSI placement and floorplanning:** Mirhoseini et al., Nature 594, 207–212 (2021), which poses chip floorplanning as an RL problem using an edge-based graph convolutional architecture. Richest analogue for geometric placement with a measurable reward.
- **Quantum circuit qubit mapping and routing:** extensive RL literature, all digital.

---

## 5. The plan

### Phase 1, weeks 1 to 6: benchmark harness (Direction C)

Deliverable: an open, reproducible harness comparing embedding strategies on common instances with common metrics. Publishable on its own. This is the floor.

1. **End-to-end toy.** Place roughly 5 atoms by hand in Pulser, write a pulse, run on the local emulator, decode bitstrings. Prove the loop works before automating anything.
2. **Feasibility oracle and constraint validator.** Encode real device limits per section 6 and reject invalid registers.
3. **Instance library.** King-lattice MIS data, Erdős-Rényi at p=0.25 and p=0.75 to match the baseline, PROTEINS via TUDataset, plus the up-to-1000-qubit MIS instance set.
4. **Embedder wrappers.** Fruchterman-Reingold, gadget reduction, GLS divide-and-conquer, and DEN if it can be run. Common interface, common scoring.
5. **Metrics and reporting.** Section 6.

### Phase 2, weeks 6 to 13: learned selector (Direction A)

Use the harness as the training and evaluation environment.

6. **Reimplement GLS faithfully.** Seed, growth, ranking rule, hyperparameters k=4, s=2, ℓ=10. Reproduce its published behaviour before attempting to beat it.
7. **Imitation learning.** Train a policy to reproduce GLS's choices. Establishes that the state representation and action space are workable.
8. **Fine-tune.** Contextual bandit or simple policy gradient first. Escalate to deep RL only if the light version shows promise.
9. **Head-to-head evaluation** on identical instances, identical lattice, identical MWIS sub-solver.
10. **Optional QPU validation** on a small final batch, if access arrives.
11. **Paper.**

### Governing constraint

Phases must not be reordered. No learning before the harness and the reproduced baseline exist. This is the primary predictable failure mode: a tuned model with nothing to compare it against.

---

## 6. Evaluation

| Metric | Definition |
|---|---|
| Independent set size / optimality gap | Against CPLEX or Gurobi and simulated annealing on the same instance |
| QPU or emulator call count | Number of subproblems dispatched to reach the solution |
| Total atoms used | Sum across all extracted subgraphs |
| Feasibility rate | Fraction of instances producing a valid embedding under hardware constraints |
| Embedding wall-clock time | Reported separately and included in any end-to-end timing |

**Beating GLS on optimality gap, or matching it at a lower call count, is a defensible novelty claim.** Either suffices.

Report the graph class, instance size, baseline used, and whether embedding time is included, for every result. Results missing any of those four are not usable in the paper.

### Instance families

The two-instance demonstration the supervisor asked for maps onto the geometric versus abstract contrast:
- **Geometric, near-UDG:** antenna or sensor placement, EV charging siting, facility location. Embedding is nearly free; baselines do well.
- **Abstract, non-geometric:** shift and crew scheduling (set packing), cardinality-constrained portfolio selection, exam timetabling. No natural coordinates; embedding is the bottleneck.

---

## 7. Hardware constraints and datasets

### Device constraints

**Verify against the installed Pulser version and the current device generation before relying on these.** Different papers report different numbers because they target different device eras. Query `Device.specs` programmatically rather than trusting any document, this one included.

Verified against Pulser 1.9.0 `AnalogDevice` on 2026-09-06: 2D, 38 µm radius from origin, 5 µm minimum neighbor separation, Rydberg principal quantum number 60, Ising interaction coefficient 865723.02, maximum sequence duration 6000 ns, and a single `rydberg_global` channel with maximum amplitude 12.566 rad/µs and maximum absolute detuning 125.664 rad/µs. These agree with arXiv:2510.25473.

**Layouts, and why free placement is not available.** `AnalogDevice` rejects freely-placed registers: the register must be carved out of a `RegisterLayout`. The decision available to a compiler is therefore which traps to occupy, not where to put atoms, which is the regime the baseline works in (section 3.4), enforced by the library rather than merely conventional.

The layout does not have to be the one Pulser ships. `AnalogDevice` carries one pre-calibrated layout, `TriangularLatticeLayout(61, 5.0µm)`, but it accepts custom layouts, and Orion Beta advertises layouts on demand between 10 and 200 traps. **Lattice size is a harness parameter, not a fixed given.**

**Atom ceiling is 80, on a large enough layout.** Two bounds apply and either can bind. `max_layout_filling` of 0.5 scales with the layout, so the 61-trap default holds only 30 atoms and rejects 31. `max_atom_num` of 80 does not scale, so it binds from roughly 160 traps upward. Verified: a custom `TriangularLatticeLayout(200, 5.0)` is accepted and holds 80; 250 traps is rejected on the 38 µm radial limit. An earlier version of this section claimed a flat 30-atom ceiling. That was the default layout mistaken for the machine. See D12, superseded by D14.

**Orion Beta, from Pasqal's own technical overview** (the closest official spec sheet; note the paper's machine is Orion *Alpha*, an earlier generation): 100 qubits maximum, 5 µm minimum atomic distance, 35 µm maximum radial distance, 0.5 maximum layout filling fraction, 10 to 200 traps, layouts on demand, 500 maximum runs per job, 6000 ns maximum sequence duration, 16 ns minimum pulse duration, 2 MHz maximum Rabi frequency, 7.75 MHz maximum detuning. Two of these disagree with Pulser's `AnalogDevice`: radial distance 35 against 38 µm, and maximum detuning 7.75 against 20 MHz. Different device generations. The Rabi maximum agrees exactly, since 2 MHz is 12.566 rad/µs.

**Shot budget: 500.** Settled. Pasqal's Orion Beta overview states 500 maximum runs per job, independently confirming arXiv:2510.25473. Pulser 1.9.0 `AnalogDevice.max_runs` reports 2000, which is the outlier and should not be designed against. 500 bounds any reward signal computed from measurement statistics.

**Emulation ceiling binds before either hardware ceiling.** QuTiP state-vector emulation costs grow as 2^n. Measured on the owner's machine with the standard 4000 ns adiabatic ramp and 100 shots: 0.06 s at 6 atoms, 0.76 s at 10, 2.2 s at 12, 8.4 s at 14, 41.7 s at 16. The step ratio per two atoms added is not constant, it worsens, running 2.9x, 3.8x, then 5.0x. At 18 atoms the run did not complete in over 11 minutes, against roughly 190 s extrapolated from the trend, so the pure-Python path collapses past 16 rather than degrading smoothly. Suspected memory-bound, not confirmed.

Practical reading: roughly 12 atoms is the ceiling for anything inside a training loop, and 16 is the ceiling for routine validation. Neither hardware ceiling, 30 on the default layout or 80 on a large one, is reachable in exact simulation on this machine. EMU-MPS or EMU-SV (section 11) are the escape hatches if larger pieces are needed, at the cost of an approximation and a benchmark run that has not been done yet. This is the constraint behind D13.

Pulser's `DigitalAnalogDevice` reports different values (2D, max 100 atoms, 50 µm from origin, 4 µm minimum spacing, Rydberg level 70, max layout filling fraction 0.5). It is a convenience object for tutorials, not a machine.

Also reported in arXiv:2605.03503: hardware may impose row-spacing requirements, for example atoms must share a row or be separated by at least 4 µm on the y axis.

### Datasets

- **KAIST king-lattice MIS dataset.** Nature Scientific Data, DOI 10.1038/s41597-024-02926-9 (2024). 582,916 Rydberg measurement events across 733,853 graphs, up to 141 atoms on the union-jack king lattice, with raw images and binary ground states. Real experimental data, free, no hardware access needed. **Build this into the harness.**
- **Up-to-1000-qubit MIS instance set**, from the benchmarking-at-scale work that also benchmarked quera_aquila to 102 qubits and pasqal_fresnel to 85.
- **Erdős-Rényi generators** at p=0.25 and p=0.75 to match the baseline exactly.
- **TUDataset / PROTEINS**, for comparability with the DEN results.

---

## 8. Verification status

### Verified against primary sources
- Rydberg blockade encodes MIS natively.
- UDG embedding is NP-hard; not all graphs admit a representation.
- Quadratic gadget overhead, N ≲ 4n(n−1).
- GLS is greedy, deterministic, linear overhead.
- Local detuning / DMM handles node weights non-geometrically.
- arXiv:2506.05536 is digital circuit compilation, not analog embedding.
- **Ising interaction coefficient 865723.02 at Rydberg level 60**, matching arXiv:2510.25473. Checked against Pulser 1.9.0 `AnalogDevice`, 2026-09-06.
- **Device constraints**: 2D, 38 µm radial extent, 5 µm minimum separation, 6000 ns maximum sequence duration, single global Rydberg channel at 12.566 rad/µs maximum amplitude. Same source and date.
- **The atom ceiling and the layout rules.** 30 on the shipped 61-trap layout, 80 on a custom 200-trap one, 250 traps rejected on radial extent. Confirmed by direct construction and rejection. See section 7, D12 and D14.
- **Maximum trap degree is 6 on a triangular lattice, at any lattice size.** Checked at 61 and 200 traps. See D15.
- **The shot budget is 500.** Stated by Pasqal's Orion Beta technical overview and independently by arXiv:2510.25473. Pulser's `max_runs` of 2000 is the outlier.
- **GLS hyperparameters k=4, s=2, ℓ=10, and the triangular lattice choice at roughly 60%.** Read directly from arXiv:2508.06130. Section 3.4 was accurate.
- **The baseline's Rydberg physics matches ours.** The paper states C6/h = 138 GHz·µm⁶, which is 0.16% from Pulser's level-60 coefficient. Blockade radii computed here transfer to the paper's setting.
- **Geometric instances are classically easy, measured, not assumed.** `ev.py`, real charging station coordinates for the Twin Cities, Los Angeles and Seattle, redundancy radii 300 m to 8 km. Classical randomised greedy reaches the exact optimum or within a fraction of a vertex at every radius on every city. At 5 km: baseline 5.4% gap, learned selector 1.6%, classical greedy 0.6%. See D19.
- **Section 6's first win condition is met; the second is not.** Final head-to-head on the benchmark library, `metrics.demo()`, 2026-09-06. Abstract family: force-directed 11.3%, GLS baseline 6.7%, classical randomised greedy 4.8%, learned selector at tuned widths **1.3%**, on effectively the same atoms (15.1 against 14.3). Geometric family: GLS 2.3%, learned selector 0.0%, matching classical greedy exactly. The cost is real and goes the wrong way on call count: 3.2 QPU calls against the baseline's 2.1, plus classical time for the extra candidates. So the claim is a smaller optimality gap, NOT the same gap at a lower call count. See D18.
- **More training data does not help.** The policy saturates at roughly 60 training states, five graphs: 5.7% at k=4 with 5 graphs and 5.7% with 90. Ranking a handful of candidates is an easy task and seven features learn it almost immediately. Record this wherever a reviewer might assume the result is data-limited.
- **Selection is nearly solved by simple means; candidate generation is the real bottleneck.** Measured 2026-09-06, reproducible via `policy.scaling()` and `gnn.demo()`. Held-out ranking accuracy: baseline rule 78.1%, linear hand features 95.3%, GNN 96.9%. Held-out optimality gap against candidates generated per round: baseline 8.7% / 7.3% / 7.9% at k = 4 / 8 / 16, learned policy 6.0% / 3.7% / 2.9%, classical randomised greedy floor 4.8%. The pattern repeats on a shifted set (n=80, p=0.15) with the policy crossing that set's 6.3% floor at k=16. See D16 and D17.
- **The subgraph-selection decision carries roughly ten points of optimality gap, and no myopic rule captures it.** Measured in this repository on 2026-09-06, reproducible via `metrics.headroom()`. On king-s9-f80, er-n20-p25, er-n50-p25 and er-n100-p25: GLS best-of-1 leaves a 10.3% mean gap, best-of-20 leaves 3.3%, and best-of-60 reaches the exact optimum on all four. Randomised min-degree greedy, given the same restart budget, saturates at 2.8% and never reaches the optimum. So the good decompositions already lie inside GLS's search space and are simply not found in one shot.

  Two hand-tuned levers were swept and neither captures it. Six ranking directions span 8.9% to 17.5% mean gap, with random ranking at 10.4% sitting near the best hand rule; the paper's own "arbitrary" random seed vertex beats both min-degree and max-degree starts (9.2% against 13.4% and 13.8%). **The failure of the local heuristics is the positive evidence for D8**, not a setback: a decision that no myopic rule captures but restarts do reach is the signature of a sequential, state-dependent problem, which is what a learned policy is for. A policy matching best-of-60 quality at best-of-1 cost is the concrete Phase 2 target.

### Not yet verified. Confirm before building on these.
- **The lattice size Cazals et al. use in arXiv:2508.06130. The paper does not state it.** The full text was read on 2026-09-06: it fixes the lattice type (triangular) and the GLS hyperparameters, but gives no trap count, spacing, or maximum subgraph size, and the largest-subgraph distribution appears only as a histogram in the Fig. 1(c) inset. Their machine is Orion Alpha, an earlier generation than the Orion Beta sheet in section 7, so its 200-trap figure is an upper bound rather than a match. Treat lattice size as a swept harness parameter and report it with every result. Resolve by reading the Fig. 1(c) inset axis, or by asking Pasqal directly if QPU contact materialises (D11).
- **The sort direction of the GLS ranking rule.** The paper ranks candidates by "a ranking strategy (e.g. degree)" and elsewhere by "vertex degree and number of neighbors already assigned", but never states which way the sort runs or how the two keys combine. `gls.DEFAULT_RANK` assumes most-constrained-first, meaning more already-placed neighbours first and lower degree as the tiebreak. Any comparison against the paper's published numbers is therefore a comparison against an assumed rule, not a reproduction. This is also the exact function D8 proposes to replace, so the ambiguity bounds how strong a "we beat GLS" claim can be: state the assumed rule alongside any such result. Resolve by finding the authors' code or asking them.
- **The restart budget behind the paper's headline comparison.** arXiv:2508.06130 reports GLS achieving a smaller optimality gap than a classical greedy baseline, but does not state how many restarts or independent runs produce its reported numbers. In this reimplementation single-shot GLS *loses* to randomised min-degree greedy (10.3% against 4.2%) and only overtakes it from roughly best-of-20 onward. Either the paper's figures assume a restart budget it does not report, or its greedy baseline is weaker than min-degree greedy, which is a strong heuristic rather than a strawman. Resolve before citing their comparison, and state the restart budget alongside any result produced here. Raise with the supervisor early, since a referee will ask.
- The 76% PROTEINS figure and its exact conditions (arXiv:2605.03503).
- GLS's precise optimality-gap values at n=500. These appear in a figure rather than a table, so extract them from the plot or appendix directly.
- GAGE's exact scope and results (arXiv:2412.14976).
- Several key papers carry 2026 arXiv identifiers whose primary PDFs were not always reachable during the review: 2602.15487, 2602.17991, 2603.28933, 2605.03565, 2605.03503, 2605.04736, 2605.16944, 2607.25834, 2609.01248. Treat figures from these as provisional.

### Negative finding, strong but not exhaustive
No published RL or learned-policy work on analog Rydberg register embedding or subgraph selection. This came from a bounded search including a dedicated deep pass. Re-run a fresh literature check at project start, and again before submission.

---

## 9. Non-goals and rejected directions

### Claims this project will not make
- **That quantum beats classical.** It will not. At 100 to 250 binary variables, tuned commercial solvers are faster by large margins. Better selection widens the set of loadable problems; it does not change solve time.
- **That quantum advantage has been demonstrated.** As of September 2026 it has not been, for combinatorial optimization, on any platform. The literature calls it a candidate, not an achievement.
- **That embedding is free.** Embedding is itself NP-hard. Any end-to-end timing must include it. State this before a referee does.
- **That analog results carry guarantees.** Output is a sampled distribution, not a certified optimum. arXiv:2603.28933 names the absence of performance guarantees as a core limitation alongside layout constraints.

### Rejected, with reasons (see decision log)
- Free atom placement as the contribution: contested by DEN and GAGE (D6).
- Spectral-gap-optimized embedding as the primary thrust: exponential evaluation cost, partially occupied (D7). Optional stretch only.
- Gadget-based exact reduction: provably quadratic, infeasible at useful sizes.
- Full agentic end-to-end build including embedding-strategy selection: the slot is genuinely open, but the scope exceeds one semester. Revisit only if Phase 2 finishes early.
- Pulse-level optimal control: crowded (arXiv:2605.04628, 2602.17991, 2605.16944) and physics-heavy. Hold the adiabatic ramp fixed so selection is the only varying factor.
- Digital circuits, error correction, logical qubits, quantum chemistry: out of scope entirely.

---

## 10. Open questions

1. **Problem pair. Geometric half chosen and built; abstract half still open.** EV charging siting is implemented in `ev.py` on real OpenStreetMap coordinates for three metros, and D19 records what it demonstrates. The abstract half is not built. Leading candidates remain crew scheduling (set packing) and cardinality-constrained portfolio selection; the finance option suits the owner's background and was explicitly opened up by the supervisor. **This is the half that carries the method claim**, since D19 shows the geometric half cannot, so choose it deliberately rather than by convenience.
2. **Paper framing and venue.** Method-first (learned selection policy; quantum software or ML venue) versus application-first (real optimization on quantum hardware; operations research venue). The supervisor sent a reference paper as a quality model; match it. Ask him which one.
3. ~~**Policy architecture.**~~ **Closed by D17, 2026-09-06.** Subgraph-level action space, seven hand features over the branch state, linear scorer. A GNN encoder following the CHARME template was built and measured: 1.6 points of ranking accuracy, nothing end to end. REINFORCE on top of it changed nothing. The original text is kept below because the reasoning that led to the trap-level versus subgraph-level split is still the reasoning that makes the choice defensible.

   Original framing. State representation (graph features, partial-assignment encoding, lattice occupancy), and whether to use a GNN encoder following the CHARME template. The action space question is now sharper than "which node versus which branch", because the calibrated layout (section 7) fixes the atom positions: the policy never emits coordinates. Two candidate framings:
   - **Trap-level.** An action assigns one problem vertex to one lattice trap, and an episode builds a piece placement site by site. Fine-grained, directly mirrors what GLS does internally, and gives dense credit assignment. Cost: long episodes, and most actions are forced or invalid, so the policy spends capacity learning feasibility that the validator already knows.
   - **Subgraph-level.** An action selects the next candidate subgraph or the next seed and the growth is left to the existing GLS machinery. Short episodes, small action space, and it isolates exactly the decision D8 names as the contribution. Cost: sparse reward, and it inherits whatever ceiling GLS's growth rule imposes.
   Trap-level is the more general claim, subgraph-level is the more defensible one on a one-semester budget and matches D10's imitation-first path more naturally. Decide before step 7, not during it.
4. ~~**MWIS sub-solver.**~~ **Closed by D13, 2026-09-06.** Classical sub-solver for training, emulator and QPU for validation only. The measured emulation cost (section 7) settled it: 8.4 s per 14-atom run cannot support a training loop needing thousands of rollouts. The reframing this implies, a classical decision policy for quantum embedding, is still novel and still the D8 slot. What remains is the narrower choice of exact versus heuristic classical solver, which is a Phase 1 implementation detail, not an open direction.
5. **How to present the agentic layer** to the supervisor given arXiv:2607.25834 exists.

---

## 11. Stack

| Component | Tool |
|---|---|
| Register and pulse construction | Pulser (`pasqal-io/Pulser`) |
| High-level analog layer | QoolQit |
| MIS reference | `pasqal-io/maximum-independent-set` |
| QUBO reference | `pasqal-io/qubo-solver` |
| Local emulation | Pulser QuTiP backends; EMU-SV (state vector); EMU-MPS (GPU, 60 to 100 qubits), see arXiv:2510.09813 |
| Cloud submission | `pasqal-io/pasqal-cloud`; also Azure, Google Cloud, OVHcloud, Scaleway |
| Classical baselines and ground truth | NetworkX (Fruchterman-Reingold), CPLEX or Gurobi, simulated annealing |
| Learning | PyTorch; torch_geometric |
| Graph ML kernel (optional) | `pasqal-io/quantum-evolution-kernel` |

---

## 12. References

**Primary baseline**
- arXiv:2508.06130, Cazals, Sorondo, Onofre, Dalyac, da Silva Coelho, Vitale (Pasqal). Quantum optimization on Rydberg-atom arrays with arbitrary connectivity: gadget limitations and a heuristic approach. Phys. Rev. A 112, 062416 (2025-12-08).

**Embedding and compilation**
- arXiv:2412.14976, Quantum compilation toolkit for Rydberg atom arrays (GAGE, lazy clique reduction). Phys. Rev. Research 7, 033107.
- arXiv:2209.03965, Nguyen et al. Quantum optimization with arbitrary connectivity using Rydberg atom arrays. PRX Quantum 4, 010316.
- arXiv:2605.03565, Neural optimization for quantum architectures: DEN and ELF.
- arXiv:2605.03503, Harnessing DEN models for quantum computing tasks on neutral atom QPUs.
- arXiv:2605.04736, Neural-powered unit disk graph embedding.
- arXiv:2602.15487, Drone delivery packing problem (Fruchterman-Reingold usage).
- arXiv:2109.03517, Rydberg quantum wires.
- arXiv:2209.05164, MIS embedding for non-local bounded-degree graphs using 3D arrays.
- arXiv:2210.05604, Rydberg-blockade-based parity quantum optimization. PRL 130, 220601.
- arXiv:2410.03902, Quantum optimization with globally driven neutral atom arrays (anchor atoms).
- arXiv:2508.18147, molecular docking follow-up reusing GLS.

**Spectral gap and hardness**
- arXiv:2411.04645, Bombieri et al. PRX Quantum 6, 020306. Gadget geometry and exponential gap closing.
- arXiv:2502.04291, Cazals et al. Hard instance generation via density and treewidth.
- arXiv:2602.17991, ADGLB, spectral-gap-guided schedule.
- arXiv:2605.16944, Hamiltonian engineering for adiabatic MIS.
- arXiv:2609.01248, Backend-agnostic MWIS kernel (gadgets closing the gap).

**Hybrid and decomposition**
- arXiv:2402.05748, Benders decomposition with neutral atoms. Phys. Rev. A 110, 012434.
- arXiv:2503.03518, enhanced Benders (feasibility 68% to 100%, optimality 52% to 86%).
- arXiv:2603.28933, LP-Quts iterative optimization with partial convergence guarantees.
- arXiv:2510.18540, grid-partitioned MWIS.

**RL, transferable**
- arXiv:2506.05536, QC-Daemon. RL for digital compilation on reconfigurable atom arrays. Near-collision, not a blocker.
- arXiv:2406.07124, CHARME. GNN plus RL for D-Wave minor embedding.
- arXiv:2507.16004, Nembrini et al. PPO for minor embedding. Quantum Machine Intelligence, DOI 10.1007/s42484-026-00341-4.
- Mirhoseini et al. A graph placement methodology for fast chip design. Nature 594, 207–212 (2021).

**Detuning and hardware**
- arXiv:2510.25473, Detuning choice for solving MIS and MWIS. Device constraints and DMM.
- PRX Quantum 6, 010301, weighted graph optimization via local light shifts.
- arXiv:2510.09813, emu-sv and emu-mps emulators.

**Foundations and context**
- Ebadi et al. Quantum optimization of maximum independent set using Rydberg atom arrays. Science 376, 1209 (2022).
- arXiv:2403.11931, Dalyac et al. Graph algorithms with neutral atom quantum processors. Eur. Phys. J. A 60, 177 (2024).
- Kim, Kim, Park, Byun, Ahn. King-lattice MIS dataset. Nature Scientific Data, DOI 10.1038/s41597-024-02926-9 (2024).

**Agentic and autoformulation**
- arXiv:2607.25834, Pasqal agentic workflow (Dalyac, Dauphin, Henriet).
- arXiv:2509.00099, LLM-QUBO.
- OptiMUS, ORLM, LLMOPT, NL4Opt lineage.

**Adjacent**
- arXiv:2107.03247, Quantum evolution kernel.

---

## 13. Notes for agents

- Verify device specs and library APIs against the installed version. Values here drift.
- Never introduce claims about quantum speedup into generated text, README files, or paper drafts. See section 9.
- Preserve the distinction between feasibility (can the graph be embedded at all) and quality (how good is the embedding and the resulting solution). Conflating them produces misleading numbers.
- Do not reorder the phases in section 5.
- When adding a claim sourced from a 2026-dated preprint, add it to the unverified list in section 8 rather than stating it flatly.
- **Bitstring decoding: Pulser indexes bits by the register's insertion order, never by sorted atom names.** `sorted(reg.qubits)` agrees with insertion order only up to nine atoms, then diverges, because "q10" sorts before "q2". Past that point every decoded solution is silently wrong while still looking like a plausible independent set. Always decode against `list(reg.qubits)`. This bug was found and fixed in `toy_mis.py`; do not reintroduce it.
- Writing preference: no em dashes anywhere in generated prose.
