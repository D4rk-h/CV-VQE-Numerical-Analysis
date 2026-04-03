import numpy as np


def H_BoseHubbard(samples, N, t=1, U=1, mu=1):
    """
    Weyl symbol of Bose-Hubbard Hamiltonian, expressed in terms 
    of the quadratures x and p, (for abreviations, we call α = (x+ip)/sqrt(2), 
    and α^* = (x-ip)/sqrt(2)), is defined as:

        H = ∑​_i [-t(α^*_i α_i+1 + α^*_i α_i+1) + (U|α_i|^4)/2 - (μ + U)|α_i|^2 + (μ/2 + 3U/8)]

    where t is the hopping parameter, U is the on-site interaction strength, and μ is the chemical potential.

    Following the Stornati et al. paper, U=t=mu=1, so we can simplify the expression to:
        H = ∑​_i [-(α^*_i α_i+1 + α^*_i α_i+1) + (|α_i|^4)/2 - 2|α_i|^2 + 7/8]
    
    samples: shape (n_samples, 2N)
    x: shape (n_samples, N)
    p: shape (n_samples, N)

    """
    x = samples[:, :N]
    p = samples[:, N:]
    r2 = x**2 + p**2
    hopping = np.sum(x * np.roll(x, -1, axis=1) + p * np.roll(p, -1, axis=1),axis=1)
    interaction = np.sum((r2**2) * U / 2, axis=1)
    chemical = np.sum(r2 / 2, axis=1) * (mu + U)
    constant = N * (mu/2 + 3*U/8)
    return -t * hopping + interaction - chemical + constant