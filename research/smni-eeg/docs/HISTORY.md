# History — Ingber's SMNI program, the EEG dataset, and the calcium line

## 0. The most interesting thread: estimating molecular Ca²⁺ activity from EEG

Ingber's **recent** work (2012→present) is the headline interest for this project. It is a
genuinely unusual **physics** move: treating the *macroscopic* EEG field as a top-down driver
of *molecular-scale* calcium dynamics, and quantifying the coupling.

**The mechanism.** Macrocolumnar neuronal firings (described by SMNI, below) are recast as a
**magnetic vector potential `A`**. Free regenerative **Ca²⁺ waves** — prominent in synaptic
and extracellular space, including at **tripartite (neuron–glia–neuron / astrocyte) synapses**
— are charged particles, so their dynamics obey the **canonical momentum**

> **Π = p + qA**   (SI units), with Ca²⁺ charge **q = −2e**

Ingber's calculations show the EEG-derived `A` is large enough to materially influence the
Ca²⁺ momentum `p`, in **both classical and quantum** treatments. The quantum version propagates
a Ca²⁺ wave-packet path integral (**qPATHINT**, with a tree variant **qPATHTREE**) and couples
it back to the classical SMNI dynamics across scales. Ingber is candid that the quantum effects
are **speculative**, but reports the calculations are consistent with experimental EEG/STM data.

**Why it matters here:** it makes the *same* EEG recordings we hold (the SMNI_CMI alcoholism
set) a substrate for a multi-scale model that reaches from scalp potentials down to ion-channel
/ neurotransmitter-release physics — i.e. "reading out" a molecular variable (Ca²⁺ momentum)
from electrophysiology.

### Key calcium-line papers
| Year | Title | Venue | Locators |
|---|---|---|---|
| 2014 | Electroencephalographic field influence on calcium momentum waves (Ingber, Pappalepore, Stesiak) | *J. Theoretical Biology* 343:138–153 | arXiv:1105.2352 · PubMed 24239957 · SSRN 2187029 |
| 2015 | SMNI: Large-scale EEG influences on molecular processes | *J. Theoretical Biology* | arXiv:1206.6286 · SSRN 2691682 |
| 2018 | Quantum Calcium-Ion Interactions with EEG | *Sci* (MDPI) 1(1):20 | mdpi.com/2413-4155/1/1/20 |
| 2020 | Quantum Calcium-Ion Affective Influences Measured by EEG | preprint/SSRN | SSRN 3698109 · ingber.com/quantum20_affective.pdf |

## 1. The supercomputer projects

The calcium/quantum modeling was developed under **HPC allocations**, which is why it scaled:

- **XSEDE** allocations from **Feb 2013**: **PHY130022**, then **TG-MCB140110**.
- **TG-MCB140110** — *"Quantum path-integral qPATHTREE and qPATHINT algorithms"* — ran **2017**,
  renewed through **Dec 2018**.
- Earlier compute on **Trestles** at the **San Diego Supercomputer Center (SDSC), UC San Diego**.
- These allocations produced the qPATHINT/qPATHTREE codes and the multi-scale (molecular ↔
  columnar ↔ regional) fits. (Our compute analog today is the **DGX Spark** on the LAN.)

## 2. The foundation: Statistical Mechanics of Neocortical Interactions (SMNI)

Starting ~1981 and formalized through the 1990s, Ingber derived aggregate behavior of cortical
**macrocolumns** from the statistical electrical-chemical properties of synaptic interactions,
producing a **nonlinear, nonequilibrium statistical mechanics** of EEG expressed as **path
integrals of multivariate conditional probabilities** (Lagrangian formulation). Fits were done
by **Adaptive Simulated Annealing (ASA)**, Ingber's own global optimizer.

### Foundational bibliography
| Year | Title | Venue |
|---|---|---|
| 1989 | Very fast simulated re-annealing (origin of ASA) | *Mathl. Comput. Modelling* 12(8):967–973 |
| 1991 | SMNI: A scaling paradigm applied to electroencephalography | *Phys. Rev. A* 44(6):4017–4060 |
| 1997 | **SMNI: Canonical momenta indicators of EEG** | *Phys. Rev. E* 55(4):4578–4593 (arXiv:physics/0001052) |
| 1998 | SMNI: Training and testing canonical momenta indicators of EEG | *Mathl. Comput. Modelling* |
| 2009 | SMNI: Nonlinear columnar electroencephalography | SSRN 1426583 |

**Canonical Momenta Indicators (CMI)** are the feature set from the 1997 PRE paper: momenta
conjugate to the SMNI fields, used to discriminate signals (e.g. alcoholic vs control). The
*same `p + qA` momentum idea* later carries straight into the calcium line — the CMI and the
Ca²⁺ canonical momentum are the same mathematical object applied at different scales.

## 3. Dataset provenance

The `SMNI_CMI` EEG data was distributed by Ingber **with** the 1997 CMI paper, mirrored at
`hebb.uoregon.edu/~ingber/EEG_CMI/` and `ingber.com/smni_eeg_data.html` (see the dataset
`README`, preserved at `data/TRAIN/README`).

- **Origin:** acquired in **Henri Begleiter's** Neurodynamics Laboratory, **SUNY Health Science
  Center, Brooklyn**, to study **genetic predisposition to alcoholism**; also the UCI ML
  Repository "EEG Database."
- **Paradigm:** visual object-recognition ERP. Stimuli are **Snodgrass & Vanderwart (1980)**
  line drawings; conditions **S1 (single stimulus)**, **S2 match**, **S2 nomatch**.
- **Groups:** **alcoholic** (`co*a*`) vs **control** (`co*c*`).
- Format and per-set counts: see [`DATASET.md`](DATASET.md).

## 4. Reproductions on this dataset

- **Wang, Begleiter & Porjesz (2000)** — *"Trilinear Modeling of Event-Related Potentials,"*
  *Brain Topography* 13(1) (PubMed 10912734). The **Topographic Component Model (TCM)** writes
  ERPs as a trilinear (space × time × subject/condition) decomposition — a tensor / PARAFAC-style
  model. **Already reproduced on the DGX Spark** (this project will fold that in / re-run it).

---
*Note on rigor:* Ingber himself flags the quantum-calcium effects as speculative; the classical
`p + qA` coupling and SMNI/CMI fits are the better-established parts. Treat the qPATHINT
consciousness claims as a hypothesis to test against data, not settled result.
