from math.gaussian import build_wigner
from math.polynomial import build_S
import numpy as np


def compute_variational_ansatz_layered(xi, dx, dp, n_modes, O, Z, L):
    """
    Like compute_variational_ansatz, but applies exactly L ladder steps
    and returns a list of intermediate Wigner functions W_list[0..L].
    W_list[0] = Gaussian baseline, W_list[ℓ] = after ℓ ladder applications.
    """
    W_G = build_wigner(xi, O, Z, n_modes)
    W_list = [W_G.copy()]

    W = W_G.copy()
    for j in range(L):
        mode_idx = j % n_modes
        S_k = build_S(O, xi, mode_idx)
        weight = np.abs(S_k) ** 2

        W_unnorm = weight * W
        K = np.sum(W_unnorm) * dx * dp
        if np.abs(K) < 1e-300 or not np.isfinite(K):
            W_list.append(W.copy())
            continue
        W = W_unnorm / K
        W_list.append(W.copy())

    return W_list