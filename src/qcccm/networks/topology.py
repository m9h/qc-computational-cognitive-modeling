"""Network topologies for multi-agent quantum cognitive networks."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array


class NetworkTopology(NamedTuple):
    """Graph topology for a multi-agent quantum network."""

    adjacency: Array  # (n_agents, n_agents)
    n_agents: int
    labels: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Standard topologies
# ---------------------------------------------------------------------------


def complete_graph(n: int) -> NetworkTopology:
    """Fully connected graph (all-to-all interaction)."""
    adj = jnp.ones((n, n)) - jnp.eye(n)
    return NetworkTopology(adjacency=adj, n_agents=n)


def ring_graph(n: int) -> NetworkTopology:
    """Ring topology: each agent connected to two nearest neighbours."""
    adj = jnp.zeros((n, n))
    idx = jnp.arange(n)
    adj = adj.at[idx, (idx + 1) % n].set(1.0)
    adj = adj.at[idx, (idx - 1) % n].set(1.0)
    return NetworkTopology(adjacency=adj, n_agents=n)


def star_graph(n: int) -> NetworkTopology:
    """Star topology: agent 0 is the hub, connected to all others."""
    adj = jnp.zeros((n, n))
    adj = adj.at[0, 1:].set(1.0)
    adj = adj.at[1:, 0].set(1.0)
    return NetworkTopology(adjacency=adj, n_agents=n)


def random_graph(n: int, p: float, key: Array) -> NetworkTopology:
    """Erdos-Renyi random graph G(n, p).

    Args:
        n: number of agents.
        p: edge probability.
        key: JAX PRNG key.
    """
    mask = jax.random.uniform(key, (n, n)) < p
    # Symmetrise and remove diagonal
    upper = jnp.triu(mask, k=1)
    adj = (upper + upper.T).astype(jnp.float32)
    return NetworkTopology(adjacency=adj, n_agents=n)


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------


def adjacency_to_stochastic(adjacency: Array) -> Array:
    """Convert adjacency matrix to column-stochastic transition matrix.

    Adds self-loops (identity) then normalises columns to sum to 1.

    Args:
        adjacency: (n, n) adjacency matrix (non-negative).

    Returns:
        (n, n) column-stochastic matrix.
    """
    n = adjacency.shape[0]
    with_self = adjacency + jnp.eye(n)
    col_sums = jnp.sum(with_self, axis=0, keepdims=True)
    return with_self / jnp.clip(col_sums, 1e-12, None)
