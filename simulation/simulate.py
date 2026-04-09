import numpy as np
from model.math.hamiltonian import H_BoseHubbard as H_W
from model.math.polynomial import compute_P_recursive


def run_simulation(samples, V, L, N, j):
    P = compute_P_recursive(samples, V, L, j)
    H_symbols = H_W(samples=samples, N=N)
    energy = np.mean(H_symbols * P) / np.mean(P)

    print(f"__ L={L} Results __")
    print(f"Exp. Value <H>: {energy:.4f}")
    print(f"P Range: [{P.min():.2e}, {P.max():.2e}]")
    print(f"Negative Fraction: {(P < 0).mean():.4f}")

    return P, energy