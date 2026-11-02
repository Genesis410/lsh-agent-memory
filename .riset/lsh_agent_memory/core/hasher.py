"""Data-dependent hashing using PCA and Iterative Quantization (ITQ)."""

from typing import Tuple
import numpy as np


def fit_itq(
    X: np.ndarray,
    bits: int = 64,
    iters: int = 50,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit data-dependent hyperplane projection using PCA and ITQ.

    Args:
        X: Training embeddings of shape (N, d).
        bits: Target code length in bits (must be multiple of 8, e.g., 64 or 128).
        iters: Number of alternating SVD rotation iterations.
        seed: Random seed for initial orthogonal rotation matrix.

    Returns:
        Tuple of (mu, W) where mu is (d,) and W is (d, bits).
    """
    if bits % 8 != 0:
        raise ValueError(f"bits must be multiple of 8, got {bits}")
    if X.shape[1] < bits:
        raise ValueError(f"Feature dim {X.shape[1]} must be >= bits {bits}")

    mu = X.mean(axis=0, dtype=np.float64)
    Xc = X - mu
    # Covariance matrix (d, d)
    C = (Xc.T @ Xc) / len(Xc)

    # PCA via eigendecomposition of symmetric covariance
    _, V = np.linalg.eigh(C)
    # Take top `bits` eigenvectors, highest eigenvalues first
    P = V[:, -bits:][:, ::-1]  # (d, bits)
    Z = Xc @ P  # (N, bits)

    # Initialize random orthogonal rotation matrix R
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((bits, bits))
    R, _ = np.linalg.qr(G)

    # Alternating optimization: minimize ||B - Z @ R||_F^2
    for _ in range(iters):
        B = np.sign(Z @ R)
        # Avoid sign(0) == 0 causing zero bits
        B[B == 0] = 1.0
        U, _, Vt = np.linalg.svd(B.T @ Z)
        R = (U @ Vt).T

    W = (P @ R).astype(np.float32)
    # ponytail: full SVD on 64x64 or 128x128 is sub-millisecond. Upgrade path: randomized SVD if bits >= 512.
    return mu.astype(np.float32), W


def encode(
    X: np.ndarray,
    mu: np.ndarray,
    W: np.ndarray,
    chunk: int = 10_000,
) -> np.ndarray:
    """Encode float embeddings into packed uint8 bitcodes.

    Args:
        X: Input embeddings of shape (N, d).
        mu: Centering vector of shape (d,).
        W: Projection matrix of shape (d, bits).
        chunk: Chunk size to balance CPU cache and memory usage.

    Returns:
        Packed uint8 array of shape (N, bits // 8).
    """
    # ponytail: chunk=10_000 tuned for L3 cache on Ryzen 4650U. Upgrade path: dynamic chunking if d > 1024.
    out_list = []
    for i in range(0, len(X), chunk):
        batch = X[i : i + chunk]
        proj = (batch - mu) @ W
        bits = proj > 0
        packed = np.packbits(bits, axis=1)
        out_list.append(packed)

    if not out_list:
        return np.empty((0, W.shape[1] // 8), dtype=np.uint8)
    return np.vstack(out_list)
