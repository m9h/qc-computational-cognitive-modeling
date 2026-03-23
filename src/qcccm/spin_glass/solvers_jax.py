"""Pure-JAX classical and quantum Monte Carlo solvers for social spin glass models.

Drop-in replacements for the numpy solvers in ``solvers.py`` that support:

* ``jax.jit`` compilation of the inner MC sweep (no Python-level branching)
* ``jax.vmap`` over disorder seeds for batched parallel simulation
* ``jax.lax.fori_loop`` / ``jax.lax.scan`` instead of Python for-loops
* Proper ``jax.random.PRNGKey`` handling throughout (no ``numpy.random``)
* JAX arrays for all Trotter-decomposition bookkeeping

All public functions return the same ``SolverResult`` namedtuple defined in
``solvers.py``, so callers can swap implementations transparently.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from qcccm.spin_glass.hamiltonians import SocialSpinGlassParams
from qcccm.spin_glass.solvers import SolverResult


# ---------------------------------------------------------------------------
# JAX-native energy helper
# ---------------------------------------------------------------------------


def _energy_jax(J: Array, fields: Array, spins: Array) -> Array:
    """Classical energy  H = -0.5 sum_ij J_ij s_i s_j - sum_i h_i s_i."""
    interaction = -0.5 * jnp.sum(J * jnp.outer(spins, spins))
    field = -jnp.sum(fields * spins)
    return interaction + field


# ---------------------------------------------------------------------------
# Classical Metropolis — JAX
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("n_spins", "n_steps"))
def _metropolis_sweep(
    J: Array,
    fields: Array,
    temperature: Array,
    n_spins: int,
    n_steps: int,
    init_spins: Array,
    init_energy: Array,
    key: Array,
) -> tuple[Array, Array, Array, Array]:
    """JIT-compiled Metropolis single-spin-flip sweep.

    Returns (best_spins, best_energy, final_spins, energies_trajectory).
    """
    T = jnp.maximum(temperature, 1e-10)

    def body(carry, _unused):
        spins, energy, best_spins, best_energy, key = carry
        key, k_site, k_accept = jax.random.split(key, 3)

        # Pick a random site
        i = jax.random.randint(k_site, (), 0, n_spins)

        # Compute delta_E for flipping spin i
        # delta_E = 2 * s_i * (sum_j J_ij s_j + h_i)
        local_field = jnp.dot(J[i], spins) + fields[i]
        delta_E = 2.0 * spins[i] * local_field

        # Metropolis acceptance (no Python if/else)
        accept_prob = jnp.where(delta_E <= 0.0, 1.0, jnp.exp(-delta_E / T))
        accept = jax.random.uniform(k_accept) < accept_prob

        # Conditionally flip
        new_spin_i = jnp.where(accept, -spins[i], spins[i])
        spins = spins.at[i].set(new_spin_i)
        energy = jnp.where(accept, energy + delta_E, energy)

        # Track best
        improved = energy < best_energy
        best_energy = jnp.where(improved, energy, best_energy)
        best_spins = jnp.where(improved, spins, best_spins)

        return (spins, energy, best_spins, best_energy, key), energy

    init_carry = (init_spins, init_energy, init_spins, init_energy, key)
    (final_spins, final_energy, best_spins, best_energy, _), energies = jax.lax.scan(
        body, init_carry, None, length=n_steps
    )

    return best_spins, best_energy, final_spins, energies


def metropolis_spin_glass_jax(
    params: SocialSpinGlassParams,
    n_steps: int = 5000,
    key: Array | None = None,
    n_equilibrate: int = 1000,
) -> SolverResult:
    """JAX-native Metropolis Monte Carlo for a spin glass.

    Equivalent physics to :func:`metropolis_spin_glass` but JIT-compiled.

    Args:
        params: Spin glass parameters (includes J, fields, temperature).
        n_steps: Total MC single-spin-flip steps.
        key: JAX PRNG key. If *None*, derived from ``params.seed``.
        n_equilibrate: Steps to discard before recording trajectory.

    Returns:
        SolverResult with best configuration, energy trajectory, etc.
    """
    if key is None:
        key = jax.random.PRNGKey(params.seed)

    N = params.n_agents
    J = jnp.asarray(params.J)
    fields = jnp.asarray(params.fields)
    T = jnp.asarray(params.temperature, dtype=jnp.float32)

    # Random initial spins
    key, k_init = jax.random.split(key)
    init_spins = (2 * jax.random.bernoulli(k_init, shape=(N,)).astype(jnp.float32) - 1.0)
    init_energy = _energy_jax(J, fields, init_spins)

    best_spins, best_energy, _final_spins, energies = _metropolis_sweep(
        J, fields, T, N, n_steps, init_spins, init_energy, key,
    )

    # Slice off equilibration
    energies_post = energies[n_equilibrate:]

    return SolverResult(
        spins=best_spins,
        energy=float(best_energy),
        trajectory=None,  # collecting full spin trajectory in scan is memory-heavy
        energies=energies_post,
        method="metropolis_jax",
    )


# ---------------------------------------------------------------------------
# Batched Metropolis via vmap
# ---------------------------------------------------------------------------


def _single_seed_metropolis(
    seed: Array,
    J: Array,
    fields: Array,
    temperature: Array,
    n_spins: int,
    n_steps: int,
) -> tuple[Array, Array, Array]:
    """Run Metropolis for a single seed (vmap-friendly)."""
    key = jax.random.PRNGKey(seed)
    key, k_init = jax.random.split(key)

    init_spins = 2.0 * jax.random.bernoulli(k_init, shape=(n_spins,)).astype(jnp.float32) - 1.0
    init_energy = _energy_jax(J, fields, init_spins)

    best_spins, best_energy, _, energies = _metropolis_sweep(
        J, fields, temperature, n_spins, n_steps, init_spins, init_energy, key,
    )
    return best_spins, best_energy, energies


def batched_metropolis_jax(
    params: SocialSpinGlassParams,
    seeds: Array,
    n_steps: int = 5000,
) -> tuple[Array, Array, Array]:
    """Run Metropolis MC in parallel over multiple disorder seeds.

    Uses ``jax.vmap`` so all seeds execute as a single vectorised batch.

    Args:
        params: Spin glass parameters (J, fields, T, etc.).
        seeds: 1-D integer array of PRNG seeds, shape ``(B,)``.
        n_steps: MC steps per run.

    Returns:
        ``(best_spins, best_energies, energies)`` with leading batch dim ``B``.
        ``best_spins`` has shape ``(B, N)``, ``best_energies`` ``(B,)``,
        ``energies`` ``(B, n_steps)``.
    """
    J = jnp.asarray(params.J)
    fields = jnp.asarray(params.fields)
    T = jnp.asarray(params.temperature, dtype=jnp.float32)
    N = params.n_agents
    seeds = jnp.asarray(seeds)

    vmapped = jax.vmap(
        partial(_single_seed_metropolis, J=J, fields=fields, temperature=T,
                n_spins=N, n_steps=n_steps),
    )
    return vmapped(seeds)


# ---------------------------------------------------------------------------
# Transverse-field Path-Integral Monte Carlo — JAX
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("n_spins", "n_trotter", "n_steps_per_sweep", "n_sweeps"))
def _pimc_sweep(
    J: Array,
    fields: Array,
    temperature: Array,
    J_tau: Array,
    n_spins: int,
    n_trotter: int,
    n_steps_per_sweep: int,
    n_sweeps: int,
    init_replicas: Array,
    key: Array,
) -> tuple[Array, Array, Array]:
    """JIT-compiled path-integral Monte Carlo sweep.

    Each "sweep" consists of ``n_steps_per_sweep = N * P`` single-spin flips
    (one attempted flip per spin per replica on average). We run ``n_sweeps``
    such sweeps, tracking the physical-replica energy after each.

    Returns (best_spins, best_energy, energies_per_sweep).
    """
    P = n_trotter
    N = n_spins
    T = jnp.maximum(temperature, 1e-10)

    def inner_flip(carry, _unused):
        """Single spin-flip attempt within a sweep."""
        replicas, key = carry
        key, k_p, k_i, k_accept = jax.random.split(key, 4)

        p = jax.random.randint(k_p, (), 0, P)
        i = jax.random.randint(k_i, (), 0, N)

        # In-slice energy change (classical interactions, divided by P)
        local_field_in = jnp.dot(J[i], replicas[p]) / P + fields[i] / P
        delta_E_in = 2.0 * replicas[p, i] * local_field_in

        # Inter-slice (Trotter) coupling energy change
        p_prev = (p - 1) % P
        p_next = (p + 1) % P
        delta_E_tau = 2.0 * J_tau * replicas[p, i] * (
            replicas[p_prev, i] + replicas[p_next, i]
        )

        delta_E = delta_E_in + delta_E_tau

        accept_prob = jnp.where(delta_E <= 0.0, 1.0, jnp.exp(-delta_E / T))
        accept = jax.random.uniform(k_accept) < accept_prob

        new_val = jnp.where(accept, -replicas[p, i], replicas[p, i])
        replicas = replicas.at[p, i].set(new_val)

        return (replicas, key), None

    def outer_sweep(carry, _unused):
        """One full sweep of N*P flip attempts, then track energy."""
        replicas, best_spins, best_energy, key = carry

        key, k_sweep = jax.random.split(key)
        (replicas, _), _ = jax.lax.scan(
            inner_flip, (replicas, k_sweep), None, length=n_steps_per_sweep,
        )

        # Physical replica energy
        phys_spins = replicas[0]
        phys_energy = _energy_jax(J, fields, phys_spins)

        improved = phys_energy < best_energy
        best_energy = jnp.where(improved, phys_energy, best_energy)
        best_spins = jnp.where(improved, phys_spins, best_spins)

        return (replicas, best_spins, best_energy, key), phys_energy

    init_phys = init_replicas[0]
    init_energy = _energy_jax(J, fields, init_phys)
    init_carry = (init_replicas, init_phys, init_energy, key)

    (_, best_spins, best_energy, _), energies = jax.lax.scan(
        outer_sweep, init_carry, None, length=n_sweeps,
    )

    return best_spins, best_energy, energies


def transverse_field_mc_jax(
    params: SocialSpinGlassParams,
    n_trotter: int = 8,
    n_steps: int = 5000,
    key: Array | None = None,
    n_equilibrate: int = 1000,
) -> SolverResult:
    """JAX-native path-integral Monte Carlo for a transverse-field spin glass.

    Equivalent physics to :func:`transverse_field_mc` but fully JIT-compiled.

    Each of the ``n_steps`` "sweeps" performs ``N * P`` single-spin-flip
    attempts across Trotter replicas, matching the original implementation.

    Args:
        params: Spin glass parameters. ``transverse_field > 0`` enables
            quantum tunneling via Trotter coupling.
        n_trotter: Number of Trotter slices *P*.
        n_steps: Number of MC sweeps (each sweep = N*P flip attempts).
        key: JAX PRNG key. If *None*, derived from ``params.seed``.
        n_equilibrate: Sweeps to discard before trajectory recording.

    Returns:
        SolverResult with best configuration from the physical replica.
    """
    if key is None:
        key = jax.random.PRNGKey(params.seed)

    N = params.n_agents
    P = n_trotter
    T_val = max(params.temperature, 1e-10)
    Gamma = params.transverse_field

    J = jnp.asarray(params.J)
    fields = jnp.asarray(params.fields)
    T = jnp.asarray(T_val, dtype=jnp.float32)

    # Trotter coupling
    if Gamma > 0:
        J_tau = jnp.asarray(
            -0.5 * T_val * float(jnp.log(jnp.tanh(Gamma / (P * T_val) + 1e-15))),
            dtype=jnp.float32,
        )
    else:
        J_tau = jnp.asarray(0.0, dtype=jnp.float32)

    # Initialise replicas
    key, k_init = jax.random.split(key)
    init_replicas = (
        2.0 * jax.random.bernoulli(k_init, shape=(P, N)).astype(jnp.float32) - 1.0
    )

    best_spins, best_energy, energies = _pimc_sweep(
        J, fields, T, J_tau,
        n_spins=N, n_trotter=P, n_steps_per_sweep=N * P,
        n_sweeps=n_steps,
        init_replicas=init_replicas,
        key=key,
    )

    energies_post = energies[n_equilibrate:]

    return SolverResult(
        spins=best_spins,
        energy=float(best_energy),
        trajectory=None,
        energies=energies_post,
        method=f"transverse_field_mc_jax(P={P})",
        metadata={"n_trotter": P, "J_tau": float(J_tau)},
    )
