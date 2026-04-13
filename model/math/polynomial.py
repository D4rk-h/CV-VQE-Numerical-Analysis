import numpy as np
from model.math.gaussian import build_g_wigner, build_alpha


def compute_variational_ansatz_layered(xi, dx, dp, n_modes, O, Z, L):
    """
    Like compute_variational_ansatz, but applies exactly L ladder steps
    and returns a list of intermediate Wigner functions W_list[0..L].
    W_list[0] = Gaussian baseline, W_list[ℓ] = after ℓ ladder applications.
    """
    W_G = build_g_wigner(xi, O, Z, n_modes)
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

def compute_variational_ansatz(xi: np.ndarray, dx, dp, n_modes, O, Z):
    W_G = build_g_wigner(xi, O, Z, n_modes)

    W = W_G.copy()
    for j in range(n_modes):
        S_k = build_S(O, xi, j)
        weight = np.abs(S_k) ** 2
 
        W_unnorm = weight * W
 
        dV = (dx * dp) ** n_modes
        K = np.sum(W_unnorm) * dV
        if np.abs(K) < 1e-300 or not np.isfinite(K):
            return W_G
        W = W_unnorm / K
 
    return W

def build_S(O: np.ndarray, xi: np.ndarray, j: int) -> np.ndarray:
    """
    S_k = Sum{i}(u_ji * alpha_i + v_ji * alpha^*_i)
    """
    N = O.shape[0] // 2
    u, v = get_u_v(O, j, N)
    alpha = build_alpha(xi)
    return u @ alpha + v @ np.conj(alpha)

def get_u_v(O: np.ndarray, j: int, N: int):
    """
        u_ki = O_i,k (The top half of k-th column of O matrix)
        v_ki = O_i+N,k (The bottom half of k-th column of O matrix)
    """
    return O[:N,j], O[N:,j]