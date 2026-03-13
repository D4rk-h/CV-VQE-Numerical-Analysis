import numpy as np

def build_interferometer(theta: np.ndarray, N: int):
    O = np.ndarray((2*N, 2*N))
    for i in range(N):
        c = np.cos(theta[i])
        s = np.sin(theta[i])
        O_i = np.array([[c, -s], [s, c]])
        O[2*i:2*i+2, 2*i:2*i+2] = O_i
    return O

def build_V(theta: np.ndarray, r: np.ndarray):
    """
    O: Interferometer matrix 2N x 2N
    theta: List of N-1 beam splitter angles
    r: List of N squeezing parameters
    """
    N = len(r)
    R = np.diag([*r, *[1/ri for ri in r]])
    O = build_interferometer(theta, N)
    return O @ R @ O.T


def sample_Wigner_gaussian(xi, V):
    V_inv = np.linalg.inv(V)
    xi = np.ndarray(xi)