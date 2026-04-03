import numpy as np
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

def build_Z(r, N):
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
    #r = np.full(N, 0.5) # fixed constants for testing
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

def sample_xi(xi_mean, V, n_samples=50_000):
    """
    Sample xi from a normal distribution with mean 0 and variance 1.
    """
    grad1 = np.random.multivariate_normal(xi_mean, V/2, n_samples)

    # Uncomment to test mean and std of the generated samples
    #grad2 = np.random.multivariate_normal(xi_mean, V, n_samples)
    #print("mean 1:", np.mean(grad1, axis=0))
    #print("std 1: ", np.std(grad1, axis=0))
    #print("mean 2:", np.mean(grad2, axis=0))
    #print("std 2: ", np.std(grad2, axis=0))
    return grad1

