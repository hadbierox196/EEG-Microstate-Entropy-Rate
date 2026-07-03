"""Entropy metrics for EEG microstate transition sequences.

Implements the manuscript's primary biomarker, ACE(T) (row-averaged
conditional entropy, log base 2, NOT weighted by the stationary
distribution), plus the stationary-weighted entropy rate and occupancy
entropy as non-discriminative control metrics (see manuscript Section 2.5).
"""
import numpy as np


def compute_empirical_transition_matrix(labels, K):
    """First-order empirical transition matrix T_ij = P(s_{t+1}=j | s_t=i)."""
    T = np.zeros((K, K))
    for t in range(len(labels) - 1):
        i, j = labels[t], labels[t + 1]
        T[i, j] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T / row_sums
    return T


def compute_stationary_distribution(T):
    """Left eigenvector of T for eigenvalue 1, normalised to sum to 1."""
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    # Take the real part BEFORE taking abs — eig() can return a
    # complex-dtype array even when the eigenvector is real-valued.
    pi = np.real(eigvecs[:, idx])
    pi = np.abs(pi)
    total = pi.sum()
    return pi / total if total > 0 else np.full(T.shape[0], 1.0 / T.shape[0])


def per_row_entropy_bits(T):
    """H(row_i) = -sum_j T_ij * log2(T_ij), one value per source state i."""
    T_safe = np.clip(T, 1e-12, 1.0)
    return -np.sum(T_safe * np.log2(T_safe), axis=1)


def aggregate_conditional_entropy(T):
    """ACE(T): the manuscript's PRIMARY, discriminative biomarker.

    Row-averaged Shannon conditional entropy of the transition matrix,
    in bits (log base 2):

        ACE(T) = (1/K) * sum_i [ -sum_j T_ij * log2(T_ij) ]

    Theoretical maximum = log2(K) (2.0 bits for K=4). NOT weighted by
    the stationary distribution — see stationary_weighted_entropy_rate()
    for that (non-discriminative, per manuscript Section 2.5) alternative.
    """
    return float(np.mean(per_row_entropy_bits(T)))


def stationary_weighted_entropy_rate(T, pi):
    """H_rate(T): Cover & Thomas stationary-weighted entropy rate, bits.

    H_rate(T) = -sum_i pi_i * sum_j T_ij * log2(T_ij)

    Reported in the manuscript (Section 2.5) as a NON-discriminative
    control metric (Kruskal-Wallis p = 0.403 across HC/AD/FTD).
    """
    T_safe = np.clip(T, 1e-12, 1.0)
    return float(-np.sum(pi[:, None] * T_safe * np.log2(T_safe)))


def occupancy_entropy(pi):
    """H(pi) = -sum_i pi_i * log2(pi_i), the occupancy-imbalance control."""
    pi_safe = np.clip(pi, 1e-12, 1.0)
    return float(-np.sum(pi_safe * np.log2(pi_safe)))


def compute_all_metrics(labels, K=4):
    """Convenience wrapper returning everything needed downstream."""
    T = compute_empirical_transition_matrix(labels, K)
    pi = compute_stationary_distribution(T)
    ACE = aggregate_conditional_entropy(T)
    H_rate = stationary_weighted_entropy_rate(T, pi)
    H_occ = occupancy_entropy(pi)
    row_entropy = per_row_entropy_bits(T)
    return {
        "ACE": ACE, "H_rate": H_rate, "H_occupancy": H_occ,
        "row_entropy": row_entropy, "T": T, "pi": pi,
    }
