import numpy as np
from model.math.gaussian import build_alpha_from_xi


def build_polynomial(S_terms):
    pass

def build_S(O, xi, j):
    N = O.shape[0] // 2
    u = O[:N, j]
    v = O[N:, j]
    alpha = build_alpha_from_xi(xi)
    return u @ alpha + v @ np.conj(alpha)

