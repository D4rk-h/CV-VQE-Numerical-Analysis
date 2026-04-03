from collections import defaultdict
import numpy as np


def update_coeffs(coeffs, V_inv, N):
    """
    One layer of the recursion.
    
    coeffs: dict {(m,n): c_mn} where P = sum c_mn * alpha1^m * alpha1*^n
    V_inv : inverse covariance matrix (2N, 2N)
    N     : number of modes
    
    The beta term from V^{-1} contributes as linear shifts in alpha1, alpha1*.
    Specifically:
        beta = (1/2) Re[alpha1* * (V^{-1}xi)_{alpha1}]
             = (1/2) Re[alpha1* * (V_inv[0,0] + i*V_inv[0,N])/sqrt(2) * xi_1 + ...]
    
    Since V^{-1}xi is linear in xi, and xi maps to alpha via
        x1 = (alpha1 + alpha1*)/sqrt(2)
        p1 = (alpha1 - alpha1*)/(i*sqrt(2))
    beta introduces shifts of degree (m,n) -> (m+1,n) and (m,n+1).
    
    We encode V contributions as two complex coefficients:
        v_plus  = (V_inv[0,0] + i*V_inv[N,0]) / 2   (multiplies alpha1*)
        v_minus = (V_inv[0,0] - i*V_inv[N,0]) / 2   (multiplies alpha1)
    """
    v11  = V_inv[0, 0]
    v1N  = V_inv[0, N]
    vN1  = V_inv[N, 0]
    vNN  = V_inv[N, N]
    A = (v11 + vNN) / 4
    B = (v11 - vNN + 2j*v1N) / 4
    
    new_coeffs = defaultdict(complex)

    for (m, n), c in coeffs.items():
        new_coeffs[(m+1, n+1)] += c
        new_coeffs[(m, n)]     += 0.5 * c
        new_coeffs[(m+1, n+1)] -= A * c
        new_coeffs[(m+2, n)]   -= B * c
        new_coeffs[(m, n+2)]   -= np.conj(B) * c

        if n > 0:
            new_coeffs[(m+1, n-1)] += (n / 2) * c

        if m > 0:
            new_coeffs[(m-1, n+1)] -= (m / 2) * c

    return new_coeffs

def build_P(V, L, N):
    """
    Build polynomial P_2L as coefficient dictionary.
    
    V: covariance matrix (2N, 2N)
    L: number of layers
    N: number of modes
    
    returns coeffs: dict {(m,n): c_mn}
    """
    V_inv = np.linalg.inv(V)

    coeffs = defaultdict(complex)
    coeffs[(0, 0)] = 1.0

    for l in range(L):
        coeffs = update_coeffs(coeffs, V_inv, N)

    return coeffs

def compute_P_recursive(samples, V, L, j=0):
    """
    Compute the Polynomial of degree 2L, P_2L, using the recursive method.
    """
    N = V.shape[0] // 2
    V_inv = np.linalg.inv(V)

    A_j = np.zeros((2*N, 2*N))
    A_j[j, j] = 1.0
    A_j[j+N, j+N] = 1.0
    
    M = 0.5 * V_inv @ A_j @ V_inv
    tr1 = 0.5 * np.trace(V_inv @ A_j)
    tr2 = np.trace(A_j)
    K = 1 + 0.25 * (V[j, j] + V[j+N, j+N] - 2)
    
    quad = np.einsum('ni,ij,nj->n', samples, M, samples)
    P_base = (quad.real + tr1 + tr2) / K
    
    P_final = np.ones(len(samples))
    for l in range(1, L + 1):
        P_final = P_final * (P_base - (l-1)/K) 
        
    return P_final