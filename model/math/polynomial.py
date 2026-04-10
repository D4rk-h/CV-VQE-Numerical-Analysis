import numpy as np
from model.math.gaussian import build_g_wigner, build_alpha, unpack_params
from model.math.hamiltonian import H_BoseHubbard
from model.math.moyal import moyal_product


def compute_variational_ansatz(xi: np.ndarray, dx, dp, n_modes, O, Z):
    W_G = build_g_wigner(xi, O, Z, n_modes)

    W = W_G.copy()
    for j in range(n_modes):
        S_k = build_S(O, xi, j)
        weight = np.abs(S_k) ** 2
 
        W_unnorm = weight * W
 
        K = np.sum(W_unnorm) * dx * dp
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
