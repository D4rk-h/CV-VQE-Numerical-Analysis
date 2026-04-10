import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group

def build_U(N):
    """
    Build U as a random unitary matrix of size N x N, 
    sampled from the Haar measure.
    """
    return unitary_group.rvs(N)

def build_O(N):
    """
    O is defined using the Stornati et al definition:

        O ∈ Sp2N,R ∩ SO(2N) ≅ U(N) 
    
    which represents a passive transformation that commutes with the 
    operator n = a†a.

    Is a 2N x 2 symplectic orthogonal matrix that acts as a passive optical linear network.
    So:

        O = [[Re(U), -Im(U)],
             [Im(U), Re(U)]]

    where U is the N x N unitary matrix that represents the transformation.
    """
    U = build_U(N)
    return np.block([[np.real(U), -np.imag(U)], [np.imag(U), np.real(U)]])

def build_Z(N):
    """
    Z is the squeezing matrix, defined as:

        Z = diag(r1, r2, ..., rN,1/r1, 1/r2, ..., 1/rN)

    where r_i is a squeezing parameter (0<=r_i<=1), and 
    squeezes mode i at x by a factor of r_i and antisqueezes
    at p by a factor of 1/r_i.

    Now to define r, since if we do an uniform distribution 
    for r values (e.g [0,1]) most of the interval are very little
    squeezing factors (near zero squeezing in a lot of initializations),
    we start from the definition of r in decibels (following the convention
    of Stornati et al):
        
        rdB=-10log10(r2)=-20log10(r)
    
    solving for r, we get:
        
        r=10^(-rdB/20)

    """
    r_db = np.random.uniform(0, 10, N)
    r = 10 ** (-r_db / 20)
    return np.diag([*r, *[1/ri for ri in r]])

def build_cov(O, Z):
    """
    Initial state is Vacuum.
    Covariance matrix of the state, is defined as:
        V = O Z^2 O^T
    where O is the passive transformation and Z is the squeezing matrix.
    """
    return O @ Z @ Z.T @ O.T

def build_displacement(N):
    """
    Since in Stornati they explicitly say that the 
    mean value of the quadratures is zero.
    """
    return np.zeros(2*N)

def build_g_wigner(xi: np.ndarray, O, Z, N):
    V = build_cov(O, Z) + 1e-8 * np.eye(2 * N)
    V_inv = np.linalg.inv(V)
    V_det = np.linalg.det(V)
    if not np.isfinite(V_det) or V_det <= 0:
        return np.zeros(xi.shape[1])
    term1 = 1.0 / ((np.pi ** N) * np.sqrt(V_det))
    term2 = -np.einsum('ij,ij->j', xi, V_inv @ xi)
    return term1 * np.exp(np.clip(term2, -500, 0))

def build_alpha(xi: np.ndarray):
    """
    xi is expected to be a vector of shape (2*N,)
    Returns a vector of complex amplitudes alpha of shape (N,)
    """
    N = xi.shape[0] // 2
    x = xi[:N]
    p = xi[N:]
    return (x + 1j * p) / np.sqrt(2)

def _skew_hermitian_to_unitary(s: np.ndarray, N: int) -> np.ndarray:
    """
    Build U ∈ U(N) via U = exp(S) where S is skew-Hermitian.
 
    Parameter layout in s (length = N*(N-1)/2 + N = N²  real numbers total,
    which correctly counts dim U(N) = N²):
      s[0 : N*(N-1)//2]   — real off-diagonal entries  S[i,j] = -S[j,i]
      s[N*(N-1)//2 : ]    — diagonal imaginary entries  S[i,i] = i * s[...]
    """
    n_off = N * (N - 1) // 2   # number of off-diagonal pairs
    s_re  = s[:n_off]           # real off-diagonal
    s_im  = s[n_off:2*n_off]    # imaginary off-diagonal
    s_diag = s[2*n_off:]        # imaginary diagonal
 
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
    """Number of real parameters to describe U(N): N*(N-1)/2 + N = N²."""
    return N * N
 
 
def unpack_params(params: np.ndarray, n_modes: int):
    """
    Map flat optimizer params → valid symplectic matrices (O, Z).
 
    Layout (total length = n_modes + n_modes²):
      params[:n_modes]   — log-squeezing log(r_i); r_i = exp(·) > 0
      params[n_modes:]   — full U(N) via skew-Hermitian expm (N² params)
 
    Z = diag(r_1,…,r_N, 1/r_1,…,1/r_N)
    O = [[Re U, -Im U], [Im U, Re U]]  with U = exp(S), S skew-Hermitian
    """
    N = n_modes
    r = np.exp(np.clip(params[:N], -5, 5))
    Z = np.diag(np.concatenate([r, 1.0 / r]))
 
    S = _skew_hermitian_to_unitary(params[N:], N)
    U = expm(S)
    O = np.block([[np.real(U), -np.imag(U)],
                  [np.imag(U),  np.real(U)]])
    return O, Z