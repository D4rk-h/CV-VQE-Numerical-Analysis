import numpy as np


def moyal_product(f, gaussian_wigner, dx, dp, n_modes):
    """
     Since S_k is linear in xi the moyal product ends at first order
    """
    w_new = f * gaussian_wigner + (1j/2) * poisson_bracket(f, gaussian_wigner, dx, dp, n_modes)
    return w_new

def poisson_bracket(f, g, dx, dp, n_modes):
    pb = np.zeros_like(f)
    
    for i in range(n_modes):
        df_dxi = np.gradient(f, dx, axis=2*i)
        df_dpi = np.gradient(f, dp, axis=2*i + 1)
        
        dg_dxi = np.gradient(g, dx, axis=2*i)
        dg_dpi = np.gradient(g, dp, axis=2*i + 1)
        
        pb += (df_dxi * dg_dpi - df_dpi * dg_dxi)
        
    return pb