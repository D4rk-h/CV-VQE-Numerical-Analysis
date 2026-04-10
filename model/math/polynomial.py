import numpy as np


def build_S(O: np.ndarray, alpha: np.ndarray, j: int) -> np.ndarray:
    """
    S_k = Sum{i}(u_ji * alpha_i + v_ji * alpha^*_i)
    """
    N = O.shape[0] // 2
    u, v = get_u_v(O, j, N)
    S_k = ((u @ alpha) + (v @ np.conj(alpha)))
    return S_k


def get_u_v(O: np.ndarray, j: int, N: int):
    """
        u_ki = O_i,k (The top half of k-th column of O matrix)
        v_ki = O_i+N,k (The bottom half of k-th column of O matrix)
    """
    return O[:N,j], O[N:,j]
