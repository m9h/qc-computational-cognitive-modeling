# Critique & future developments — Ingber's program, then and now

Deep dive into Ingber's papers (primary source: `refs/ingber/qPATHINT/smni21_hybrid.pdf`,
"Hybrid classical-quantum computing: Applications to SMNI", 2021; plus the 1997 CMI paper and
the 2014 *J. Theor. Biol.* calcium paper). Three parts: (1) what **Ingber himself** says still
needs doing, (2) where **others** see problems, (3) the map from his **1990s constraints to
modern tooling** — the better solutions he wished he had.

---

## 1. What Ingber says still needs developing (his own words)

From `smni21_hybrid.pdf`:

1. **Port SMNI to real quantum computers.** "The author has accounts on D-Wave and Rigetti
   computers, but has *not yet ported* SMNI code to these Quantum computers." Everything to date
   is a *classical simulation* of the hybrid system on supercomputers.
2. **Realistic serial random shocks to the Ca²⁺ wave-packet.** The pre-2016 work used a
   closed-form analytic `ψ`; qPATHINT exists precisely to replace it with numerically-propagated
   wave-packets that take *shocks* as Ca²⁺ ions enter/leave during the ~hundreds-of-ms lifetime.
   This is the central unfinished modeling step.
3. **More EEG data.** Repeatedly: "Further tests of these multiple-scale models with more EEG
   data are required." Fits are per-subject and context-dependent.
4. **Region-specific spline-Laplacian.** He argues the surface-Laplacian should be applied
   *per neocortical region* (visual/auditory/somatic/abstract), not globally, "since each region
   participates in attention differently." Flagged as to-be-tested.
5. **Evolve the full SMNI Lagrangian synchronously with PATHINT** instead of the short-time
   approximation per epoch ("PATHINT could synchronously also be evolved using the SMNI
   Lagrangian").
6. **The imaginary-time phase problem (unsolved).** "After multiple foldings of the path integral,
   usually there is no audit trail back to imaginary time to extract phase information" (per a 2015
   private communication with Larry Schulman). A genuine open methodological gap for the quantum
   propagation.

## 2. His own caveats (self-identified weak points)

- **Explicitly speculative, by his own statement:** *"This particular project most certainly is
  speculative, but it is testable… This is a somewhat indirect path."*
- **Decoherence:** he concedes the Zeno / "bang-bang" coherence mechanism "may exist only in
  special contexts, since decoherence among particles is known to be very fast" (cites Preskill
  2015) — i.e. he acknowledges the standard objection up front.
- **Untested assumptions:** "assumptions… that can only be determined by future experiments."
- **Generalization quirk:** testing cost functions are *sometimes lower than training* — he reads
  this as real between-subject differences in STM strategy, but it also signals fit instability.

## 3. Where others would see problems

| Problem | Substance | Who/where |
|---|---|---|
| **Decoherence (the big one)** | Quantum coherence in a "warm, wet, noisy" brain decoheres in ~10⁻¹³–10⁻²⁰ s — far too fast to matter for ms-scale neural processing. Same objection that sank Orch-OR. | Tegmark (2000); Preskill — and Ingber's Zeno escape reads as special pleading to critics. |
| **Indirect inference** | The quantum `qA·p` contribution is tiny and inferred by *fitting EEG*, not measured. A good fit doesn't validate the mechanism — a flexible nonlinear model can fit many stories. | general modeling critique; Ingber half-concedes ("indirect path"). |
| **Identifiability / overfitting** | A nonlinear multivariate Lagrangian has many parameters; per-subject fits with train/test inversions invite overfitting worries. His defense is the **"zero-fit-parameter" philosophy** (parameters fixed to experimental ranges, only ~1 free weight for `A`) — strong, but identifiability of the rest is asserted, not proven. | standard stats/ML critique. |
| **Scale mismatch** | The SMNI dipole picture is valid only at ~mm–cm; using scalp EEG to infer molecular Ca²⁺ dynamics crosses many orders of magnitude. | EEG inverse-problem literature (Nunez). |
| **Reproducibility** | Idiosyncratic single-author C, months of CPU, no public test harness on the EEG fits → hard for others to reproduce or falsify. | (our project partly addresses this). |
| **Better-testable rivals** | Matthew Fisher's nuclear-spin **Posner molecule** proposal is a calcium-based quantum hypothesis many regard as more experimentally tractable than the EEG-field route. | Fisher (2015). |

**Net:** the *classical* SMNI/CMI layer is well-grounded (fits STM capacity 7±2 / 4±2, EEG
dispersion, Hick's law within physiological parameter ranges). The *quantum-calcium* layer is the
contested part — and Ingber agrees it's speculative-but-testable.

## 4. The 1990s constraints → modern solutions (what he wished he had)

The decisive limitation isn't theory — it's **compute and tooling**. From the paper's own
Performance section:

- **Single-core C, no parallelism.** XSEDE staff, verbatim: *"your application is not
  multi-threaded and you use single core on comet… efficiency of 1."*
- **~6 days per run.** Projected `100,000 × (0.07 + 2500×10×0.0002) = 507,000 s = 140 hr = 6
  day/run`, ×24-job arrays, ~350K SU for 100 sets.
- **ASA + Nelder–Mead simplex** as the only fitter; closed-form `ψ` hand-derived because gradients
  weren't available.

| Ingber's 1990s–2010s tool | The modern solution he lacked |
|---|---|
| Hand-derived closed-form `ψ`, analytic momenta | **Autodiff** (JAX): the path-integral likelihood is differentiable end-to-end; `Π=∂L/∂q̇` is `jax.grad`, not algebra by hand |
| ASA stochastic global search (no gradients), months of CPU | **Gradient methods** (optax/JAXopt) for the smooth part + **CMA-ES/BO** (evosax, BoTorch) only for multimodality; orders of magnitude fewer evals |
| Single-core, "efficiency of 1", 6-day runs | **GPU vectorization** (`vmap` over trials/subjects) + **GPU ADMM / operator splitting** (OSQP/SCS) where subproblems are convex → the DGX Spark does in minutes what took XSEDE days |
| Banded-matrix PATHINT kernel folding (1121-pt mesh, expensive corners) | **Differentiable SDE/PDE solvers** (`diffrax`), tensor-network / GPU kernel propagation |
| "Port to D-Wave/Rigetti" — never done | **PennyLane / Qiskit** simulators + tensor networks; hybrid VQE-style fits feasible now |
| Sparse, per-subject EEG | Large open EEG corpora + **MNE** preprocessing; we already vectorize the **FULL** 11k-trial set on a laptop |
| Single-author C, no test harness | Open Python/JAX, version control, CI, the held-out validation we already run (CMI generalizes TRAIN→TEST, t≈−11 on FULL) |

> **This is the project's thesis:** Ingber's *physics* is interesting and largely intact; his
> *implementation* was bounded by 1990s–2010s compute. Re-expressing SMNI/CMI/PATHINT in modern
> differentiable, GPU, convex-optimization terms is the contribution — not new theory, but making
> the existing theory fast, reproducible, and falsifiable.

## 5. MPPI as the teaching on-ramp: "path integrals for general computation"

Goal: give students a learning path to path integrals as a *computational* tool, not just a QFT
formalism. **MPPI (Model Predictive Path Integral control)** is the ideal first rung — it's a
working, visual, sampling-based optimizer, and was *recently shown to be ≈ preconditioned gradient
descent* (arXiv 2603.24489), tying it to optimization students already know.

A ladder (each rung is the *same idea* — sum/expectation over trajectories — in a new domain):

| Rung | Idea | Concrete artifact |
|---|---|---|
| 0 | Path integral = weighted **expectation over trajectories**, not one optimal path | toy: Brownian bridge, `exp(−S)` weighting |
| 1 | **MPPI control** — sample rollouts, weight by `exp(−cost/λ)`, update | runnable cartpole/pendulum (`pytorch_mppi`, JAX MPPI); *immediate visual payoff* |
| 2 | **PATHINT** — propagate a probability *density* via a banded Gaussian kernel `T_ij(Δt)` (Ingber eq. 12–13) | reuse our SMNI drift; 1-D → N-D histogram folding |
| 3 | **SMNI / CMI & FEP** — path integral as the **action** whose stationarity gives dynamics; canonical momenta = CMI = (in the bridge) free-energy conjugate momenta | this repo (`src/cmi.py`, `docs/PATH_INTEGRAL.md`) |
| 4 | **qPATHINT** — complex-variable / quantum path integral with serial shocks | `refs/ingber/qPATHINT`; the frontier |

The through-line: **one construct (sum over histories) unifies control (MPPI), inference (active
inference / FEP), and statistical mechanics (SMNI/PATHINT)** — which is exactly "path integrals for
general computation." MPPI gets students computing and seeing results in an afternoon; the same
mathematics then carries them up to CMI and qPATHINT.

## References
- Ingber (2021), *Hybrid classical-quantum computing: Applications to SMNI* — `refs/ingber/qPATHINT/smni21_hybrid.pdf`.
- Ingber (1997), *Phys. Rev. E* 55(4):4578 — SMNI CMI. Ingber, Pappalepore, Stesiak (2014), *J. Theor. Biol.* 343:138 — EEG→Ca²⁺.
- Tegmark (2000), *Phys. Rev. E* 61:4194 — decoherence of brain states. Fisher (2015), *Ann. Phys.* — Posner-molecule nuclear-spin proposal.
- Williams, Aldrich, Theodorou — MPPI control; "MPPI as preconditioned gradient descent" (arXiv 2603.24489).
- See `CODE_AND_REFERENCES.md` for the optimization-lineage and Active-Inference links.
