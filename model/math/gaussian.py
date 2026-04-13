import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group


def build_U(N):
    """Random N×N unitary sampled from the Haar measure."""
    return unitary_group.rvs(N)


def build_O(N):
    """
    O ∈ Sp(2N,R) ∩ SO(2N) ≅ U(N)  (passive linear-optical network).

        O = [[Re(U), -Im(U)],
             [Im(U),  Re(U)]]
    """
    U = build_U(N)
    return np.block([[np.real(U), -np.imag(U)],
                     [np.imag(U),  np.real(U)]])


def build_Z(N):
    """
    Squeezing matrix:
        Z = diag(r_1, …, r_N, 1/r_1, …, 1/r_N)

    r is sampled uniformly in dB then converted:
        r = 10^(-r_dB / 20)
    """
    r_db = np.random.uniform(0, 10, N)
    r    = 10 ** (-r_db / 20)
    return np.diag([*r, *[1.0 / ri for ri in r]])


def build_cov(O, Z):
    """Covariance matrix of the Gaussian state: V = O Z² Oᵀ."""
    return O @ Z @ Z.T @ O.T


def build_displacement(N):
    """Zero displacement (Stornati et al. convention)."""
    return np.zeros(2 * N)


def build_g_wigner(xi: np.ndarray, O, Z, N):
    V     = build_cov(O, Z) + 1e-8 * np.eye(2 * N)
    V_inv = np.linalg.inv(V)
    V_det = np.linalg.det(V)
    if not np.isfinite(V_det) or V_det <= 0:
        return np.zeros(xi.shape[1])
    term1 = 1.0 / ((np.pi ** N) * np.sqrt(V_det))
    term2 = -np.einsum('ij,ij->j', xi, V_inv @ xi)
    return term1 * np.exp(np.clip(term2, -500, 0))


def build_alpha(xi: np.ndarray):
    """
    xi : (2N, n_points)
    Returns complex amplitudes α of shape (N, n_points).
    """
    N = xi.shape[0] // 2
    x = xi[:N]
    p = xi[N:]
    return (x + 1j * p) / np.sqrt(2)


def _skew_hermitian_to_unitary(s: np.ndarray, N: int) -> np.ndarray:
    """Build skew-Hermitian S from flat params, ready for expm(S) → U(N)."""
    n_off  = N * (N - 1) // 2
    s_re   = s[:n_off]
    s_im   = s[n_off:2 * n_off]
    s_diag = s[2 * n_off:]

    S = np.zeros((N, N), dtype=complex)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            S[i, j] =  s_re[idx] + 1j * s_im[idx]
            S[j, i] = -s_re[idx] + 1j * s_im[idx]
            idx += 1
    for i in range(N):
        S[i, i] = 1j * s_diag[i]
    return S


def n_unitary_params(N: int) -> int:
    """Number of real parameters for U(N): N²."""
    return N * N


def unpack_params(params: np.ndarray, n_modes: int):
    """
    Flat optimizer params → (O, Z).

    Layout  (length = n_modes + n_modes²):
      params[:n_modes]  — log-squeezing; r_i = exp(params_i)
      params[n_modes:]  — N² params for U(N) via skew-Hermitian expm
    """
    N = n_modes
    r = np.exp(np.clip(params[:N], -5, 5))
    Z = np.diag(np.concatenate([r, 1.0 / r]))

    S = _skew_hermitian_to_unitary(params[N:], N)
    U = expm(S)
    O = np.block([[np.real(U), -np.imag(U)],
                  [np.imag(U),  np.real(U)]])
    return O, Z