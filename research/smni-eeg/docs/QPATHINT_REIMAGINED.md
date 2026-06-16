# Reimagining qPATHINT with current software

**What qPATHINT actually is (so we know what to modernize).** Ingber's qPATHINT numerically
propagates a *complex* wave-packet `ψ` for Ca²⁺ ions by **folding a banded Gaussian kernel**
`T_ij(Δt)` (the PATHINT histogram method extended to complex variables), marching in lock-step
with EEG. At each node it (a) couples the Ca²⁺ momentum `p` to the EEG-derived vector potential
`A(t)` via `Π = p + qA`, (b) applies **serial random shocks** (ions entering/leaving the packet),
and (c) takes expectations `⟨p⟩` over `ψ*ψ`. Mechanically it is **time-dependent Schrödinger /
open-system evolution with stochastic jumps** — currently implemented as single-core C, ~7500
lines, 6-day runs (see `CRITIQUE_AND_FUTURE.md`).

So the modernization question is: *what software does time-dependent, differentiable,
GPU/quantum, open-system wave-packet propagation with jumps?* Four tiers:

## A. Differentiable array (classical sim, GPU) — near-term, drop-in
- **JAX** — the kernel fold is repeated banded matrix–vector products → `jnp` + `jit` + `vmap`
  over trials/subjects. Crucially the whole propagation becomes **differentiable**, so SMNI
  params fit by gradient descent instead of ASA (kills the 6-day single-core run).
- **diffrax** (Kidger) — differentiable ODE/PDE/SDE solvers, complex-valued, adjoint gradients,
  GPU. The Schrödinger-like evolution with time-dependent `A(t)` is a natural `diffeqsolve`.
- **Krylov / expm-multiply** (sparse) — replace explicit kernel folding with matrix-exponential
  action; the banded `T_ij` is sparse → GPU sparse linalg.

## B. Open quantum systems — the *faithful* reframing of the shocks
The "ions entering/leaving the packet" *are* dissipation/measurement. Model them as such:
- **Dynamiqs** (JAX) — **best single fit**: differentiable, GPU, Lindblad + stochastic master
  equation solvers. Gives A+B together — open-system shocks *and* gradient-based fitting on the
  DGX Spark.
- **QuTiP** — mature Lindblad master equation + **Monte-Carlo wavefunction (quantum
  trajectories)**; a quantum trajectory *is* Ingber's serial-shock picture, and it keeps phase
  explicitly → sidesteps his "no audit trail back to imaginary time" phase problem.
- The regenerative Ca²⁺ wave (Zeno/"bang-bang" coherence) maps cleanly to **frequent
  measurement / dynamical decoupling** channels, which these tools express directly.

## C. Tensor networks — right structure for 1-D wave-packet + local coupling
- **quimb** (Gray) / **TeNPy** / **ITensor** — MPS + TEBD time evolution; GPU-backed. Good if we
  scale the packet's spatial dimension or couple multiple Ca²⁺ modes; shocks = local quantum
  channels / measurements between TEBD steps.

## D. Quantum-native — Ingber's unfulfilled "port to D-Wave/Rigetti", now feasible
Ingber explicitly wanted to run on real quantum hardware but never ported the code.
- **CUDA-Q (NVIDIA)** — GPU quantum simulation + hybrid classical-quantum; native to the DGX/NVIDIA
  stack we already target. Direct path to Ingber's "hybrid classical-quantum" framing.
- **PennyLane** (Xanadu) — *differentiable* quantum programming; variational/Hamiltonian
  simulation of the Ca²⁺ evolution, same gradient pipeline as tier A, runs on simulators **and**
  trapped-ion hardware via plugins.
- **Qiskit + Qiskit Dynamics** — Trotterized real-time evolution of the time-dependent
  `A(t)`-coupled Hamiltonian; execute on IBM or **Quantinuum QCCD** backends (pytket /
  qiskit-quantinuum).
- **→ The QCCD connection (the lead worth pulling):** QCCD trapped-ion machines do **mid-circuit
  measurement + reset + ion shuttling** natively. Ingber's *serial shock → partial collapse →
  regeneration* is, operationally, **mid-circuit measurement + conditional reset**. So qPATHINT's
  defining physics has a near-literal hardware primitive on QCCD — a genuinely novel mapping to
  prototype, not just a simulation.

## E. Inference reframing — ties to the Friston bridge
The *classical* PATHINT propagates a probability **density** (not amplitude): that's
message-passing / belief propagation. **RxInfer.jl** (reactive variational message passing) or a
continuous normalizing flow expresses it — linking qPATHINT's machinery to the FEP/active-inference
side (`FRISTON_BRIDGE.md`). qPATHINT (amplitudes) vs PATHINT (densities) ≈ quantum vs variational
inference of the same propagation.

## Recommended path
1. **Now:** reimplement PATHINT/qPATHINT kernel folding in **JAX** (tier A) — differentiable,
   GPU, vmap'd; immediately replaces ASA + 6-day runs. Reuse our `cmi.py` drift.
2. **Faithful physics:** move the shocks into **Dynamiqs/QuTiP** (tier B) — open-system
   trajectories, explicit phase.
3. **Frontier / novel:** prototype the **QCCD mid-circuit-measurement = serial-shock** mapping
   via **CUDA-Q / PennyLane → Quantinuum** (tier D). This is the part that would be genuinely new
   and is exactly Ingber's never-finished quantum-hardware port.

## Open question for Morgan
"qcccd repo" wasn't found locally or as a single obvious GitHub repo — confirm whether you meant
the **QCCD trapped-ion architecture** (assumed here) or a specific repository (share the link).

## References / projects
JAX · diffrax · **Dynamiqs** (github.com/dynamiqs/dynamiqs) · QuTiP · quimb/TeNPy/ITensor ·
**CUDA-Q** (NVIDIA) · PennyLane (Xanadu) · Qiskit + Qiskit Dynamics · Quantinuum QCCD
(Pino et al., *Nature* 593, 2021; arXiv:2003.01293) · RxInfer.jl. See `CODE_AND_REFERENCES.md`.
