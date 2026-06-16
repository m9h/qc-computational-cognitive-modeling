"""Generate notebooks/07_path_integrals.ipynb (the path-integral teaching ladder).

Programmatic build (nbformat) so the notebook is reproducible; execute it with
`jupyter nbconvert --to notebook --execute --inplace` to validate.
"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = []

cells.append(md(
    "# 07 — Path integrals for general computation\n\n"
    "One idea — an **expectation over trajectories**, each weighted by "
    "`exp(−action)` — reused across domains:\n\n"
    "| rung | domain | what you do |\n|---|---|---|\n"
    "| 1 | **control** (MPPI) | weight sampled rollouts, act |\n"
    "| 2 | **density propagation** (PATHINT) | fold a probability kernel |\n"
    "| 3 | **statistical mechanics / inference** (SMNI, CMI, FEP) | action → momenta |\n\n"
    "No path-integral background assumed. Each rung is the *same* construct in a new costume."))

cells.append(md(
    "## Rung 1 — MPPI: path integrals as control\n"
    "Sample many action sequences, roll them out, weight each by `exp(−cost/λ)`, and "
    "command the weighted-average action. That weighting *is* a path integral over "
    "control trajectories."))
cells.append(code(
    "import numpy as np\n"
    "rng = np.random.default_rng(0)\n\n"
    "def mppi(x0=0.0, goal=1.0, H=20, K=300, lam=0.1, steps=40):\n"
    "    u = np.zeros(H); x = x0; traj = [x]\n"
    "    for _ in range(steps):\n"
    "        noise = rng.normal(0, 0.3, (K, H))\n"
    "        xx = x + np.cumsum(u[None, :] + noise, axis=1)   # rollouts\n"
    "        costs = ((xx - goal) ** 2).sum(axis=1)\n"
    "        w = np.exp(-(costs - costs.min()) / lam); w /= w.sum()\n"
    "        u = u + w @ noise                                 # path-integral update\n"
    "        x = x + u[0]; traj.append(x); u = np.roll(u, -1); u[-1] = 0.0\n"
    "    return np.array(traj)\n\n"
    "traj = mppi()\n"
    "print('MPPI final position:', round(float(traj[-1]), 3), '(goal = 1.0)')"))

cells.append(md(
    "## Rung 2 — PATHINT: propagate a *density*\n"
    "Instead of choosing one action, fold a whole probability density forward through "
    "the short-time Gaussian kernel `T` (Ingber's PATHINT). We validate it against two "
    "analytic results: free diffusion spreads as `var = σ²·t`, and an Ornstein–Uhlenbeck "
    "drift relaxes to the stationary variance `σ²/(2k)`."))
cells.append(code(
    "import jax.numpy as jnp\n"
    "from qcccm.neuroai import pathint as pi\n\n"
    "x = jnp.linspace(-6, 6, 401)\n"
    "# free diffusion (no drift)\n"
    "T = pi.build_transition_matrix(x, pi.linear_drift(x, 0.0), diffusion=1.0, dt=0.01)\n"
    "free = pi.propagate(pi.delta_density(x, 0.0), T, 50)\n"
    "_, var = pi.moments(free[-1], x)\n"
    "print('free diffusion var :', round(float(var), 3), ' analytic σ²t =', 1.0*0.01*50)\n\n"
    "# Ornstein-Uhlenbeck relaxation g(x) = -k x\n"
    "k = 2.0\n"
    "T2 = pi.build_transition_matrix(x, pi.linear_drift(x, -k), 1.0, 0.01)\n"
    "ou = pi.propagate(pi.delta_density(x, 3.0), T2, 800)\n"
    "m, v = pi.moments(ou[-1], x)\n"
    "print('OU stationary mean :', round(float(m), 3))\n"
    "print('OU stationary var  :', round(float(v), 3), ' analytic σ²/2k =', 1.0/(2*k))"))
cells.append(code(
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "xg = np.array(x)\n"
    "for i in [0, 40, 150, 799]:\n"
    "    plt.plot(xg, np.array(ou[i]), label=f'step {i}')\n"
    "plt.legend(); plt.xlabel('x'); plt.ylabel('P(x)')\n"
    "plt.title('PATHINT: OU density relaxing to its stationary distribution')\n"
    "plt.tight_layout()"))

cells.append(md(
    "## Rung 3 — SMNI / CMI (and qPATHINT next)\n"
    "The SMNI Lagrangian `L = ½(Ṁ−g)ᵀΣ⁻¹(Ṁ−g)` defines the action; its **conjugate "
    "momenta** are the Canonical Momenta Indicators (CMI), `Π = Σ⁻¹(Ṁ−g)` — the same "
    "object the free-energy path integral uses. **qPATHINT** is rung 2 with a *complex* "
    "amplitude kernel (identical machinery, complex `T`) — the quantum step."))
cells.append(code(
    "import jax\n"
    "from qcccm.models import smni\n\n"
    "M = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 64))\n"
    "d = smni.fit_linear_drift(M)\n"
    "cmi = smni.canonical_momenta(M, d)\n"
    "print('CMI shape', cmi.shape, '= conjugate momenta  Π = Σ⁻¹(Ṁ − g)')"))

nb.cells = cells
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                              "name": "python3"},
               "language_info": {"name": "python"}}

out = os.path.join(os.path.dirname(__file__), "..", "notebooks",
                   "07_path_integrals.ipynb")
with open(out, "w") as fh:
    nbf.write(nb, fh)
print("wrote", os.path.normpath(out))
