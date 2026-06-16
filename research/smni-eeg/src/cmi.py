"""Canonical Momenta Indicators (CMI) estimation.

Background (see ../docs/PATH_INTEGRAL.md). SMNI describes the short-time evolution
of fields M(t) by a Gaussian conditional probability with Lagrangian

    L(M, Ṁ) = 1/2 (Ṁ - g(M))^T Σ^{-1} (Ṁ - g(M))

where g(M) is the deterministic *drift* and Σ the *diffusion* (noise covariance).
The **canonical momenta** conjugate to the fields are

    Π = ∂L/∂Ṁ = Σ^{-1} (Ṁ - g(M))                    <- the CMI

i.e. CMI is the (whitened) deviation of the field velocity from its modeled drift.
These are exactly the "momenta" we want to compare against the conjugate momenta of
Friston's free-energy action (the central hypothesis of this project).

This first implementation uses a **linear multivariate drift** g(M) = A·M + b (a
multivariate Ornstein–Uhlenbeck process) fit by least squares. That is the tractable
surrogate for the full nonlinear SMNI Lagrangian; the `DriftModel` interface is kept
swappable so an ASA-fit nonlinear drift can replace it without touching the CMI math.

numpy-only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FS_HZ = 256.0
DT = 1.0 / FS_HZ  # seconds per sample


def velocity(M: np.ndarray, dt: float = DT) -> np.ndarray:
    """Time derivative Ṁ via central differences along the last axis.

    M: (..., n_chan, n_samples) -> same shape.
    """
    return np.gradient(M, dt, axis=-1)


@dataclass
class CMIModel:
    """Linear-drift SMNI surrogate for canonical-momenta estimation.

    Fit on a set of trials, then `transform` any trials (e.g. fit on TRAIN, apply
    to TEST) to get per-sample CMI. `A` (drift matrix), `b` (drift bias) and the
    diffusion covariance `Sigma` are the model parameters.
    """
    A: np.ndarray | None = None        # (C, C) drift coupling
    b: np.ndarray | None = None        # (C,) drift bias
    Sigma: np.ndarray | None = None    # (C, C) diffusion covariance
    Sinv: np.ndarray | None = None     # (C, C) precision = Sigma^{-1}
    diagonal: bool = True              # use per-channel (diagonal) diffusion
    channels: list[str] | None = None
    dt: float = DT

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _design(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """From trials (N, C, T) build regression matrices.

        Returns X (samples, C) of states and V (samples, C) of velocities,
        pooled over trials and time.
        """
        V = velocity(M)                       # (N, C, T)
        # pool over trials and time -> (N*T, C)
        X = np.moveaxis(M, 1, -1).reshape(-1, M.shape[1])
        Y = np.moveaxis(V, 1, -1).reshape(-1, V.shape[1])
        ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
        return X[ok], Y[ok]

    # --- fit / transform -------------------------------------------------
    def fit(self, M: np.ndarray, channels: list[str] | None = None) -> "CMIModel":
        """Estimate drift (A, b) and diffusion Sigma from trials M (N, C, T)."""
        self.channels = channels
        X, V = self._design(M)
        C = X.shape[1]
        # least squares  V ≈ X A^T + b   ->  augment with bias column
        Xa = np.hstack([X, np.ones((X.shape[0], 1))])      # (n, C+1)
        W, *_ = np.linalg.lstsq(Xa, V, rcond=None)         # (C+1, C)
        self.A = W[:C].T                                    # (C, C)
        self.b = W[C]                                       # (C,)
        R = V - (X @ self.A.T + self.b)                     # residuals (n, C)
        if self.diagonal:
            var = R.var(axis=0)
            var[var <= 0] = np.finfo(float).eps
            self.Sigma = np.diag(var)
            self.Sinv = np.diag(1.0 / var)
        else:
            self.Sigma = np.cov(R, rowvar=False)
            self.Sinv = np.linalg.pinv(self.Sigma)
        return self

    def drift(self, M: np.ndarray) -> np.ndarray:
        """g(M) = A·M + b, applied per sample. M (N, C, T) -> (N, C, T)."""
        # einsum over channels: g[n,i,t] = A[i,j] M[n,j,t] + b[i]
        g = np.einsum("ij,njt->nit", self.A, M) + self.b[None, :, None]
        return g

    def transform(self, M: np.ndarray) -> np.ndarray:
        """Compute CMI Π = Σ^{-1}(Ṁ - g(M)) for trials M (N, C, T) -> (N, C, T)."""
        if self.A is None:
            raise RuntimeError("CMIModel must be fit() before transform()")
        V = velocity(M, self.dt)
        innov = V - self.drift(M)                           # (N, C, T)
        # Π[n,i,t] = Sinv[i,j] innov[n,j,t]
        return np.einsum("ij,njt->nit", self.Sinv, innov)

    def fit_transform(self, M: np.ndarray, channels=None) -> np.ndarray:
        return self.fit(M, channels).transform(M)

    # --- persistence -----------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(path, A=self.A, b=self.b, Sigma=self.Sigma, Sinv=self.Sinv,
                 diagonal=self.diagonal,
                 channels=np.array(self.channels or [], dtype=object), dt=self.dt)

    @classmethod
    def load(cls, path: str) -> "CMIModel":
        z = np.load(path, allow_pickle=True)
        return cls(A=z["A"], b=z["b"], Sigma=z["Sigma"], Sinv=z["Sinv"],
                   diagonal=bool(z["diagonal"]),
                   channels=list(z["channels"]), dt=float(z["dt"]))


def cmi_summary(cmi: np.ndarray) -> dict:
    """Scalar summaries of a CMI tensor (N, C, T)."""
    mag = np.sqrt((cmi ** 2).sum(axis=1))          # (N, T) momentum magnitude
    return {
        "n_trials": int(cmi.shape[0]),
        "n_chan": int(cmi.shape[1]),
        "n_samples": int(cmi.shape[2]),
        "mean_abs_cmi": float(np.nanmean(np.abs(cmi))),
        "mean_momentum_magnitude": float(np.nanmean(mag)),
        "peak_momentum_magnitude": float(np.nanmax(mag)),
    }
