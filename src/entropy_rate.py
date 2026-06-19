import numpy as np

def compute_empirical_transition_matrix(labels, K):
    T = np.zeros((K, K))
    for t in range(len(labels) - 1):
        i = labels[t]; j = labels[t+1]
        T[i, j] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T / row_sums
    return T

def compute_stationary_distribution(T):
    K = T.shape[0]
    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.abs(eigvecs[:, idx])
    pi = pi / pi.sum()
    return pi

def entropy_rate(T, pi):
    T_safe = np.clip(T, 1e-12, 1.0)
    H = -np.sum(pi[:, None] * T_safe * np.log(T_safe))
    return H

def compute_maxent_entropy_rate(labels, K=4):
    T = compute_empirical_transition_matrix(labels, K)
    pi = compute_stationary_distribution(T)
    H = entropy_rate(T, pi)
    return H, T, pi
```
