import numpy as np

"""
for the moyal non linear S product 
use Bopp shift
"""


def moyal_linear(f, gaussian_wigner, dx, dp, n_modes):
    """
     Since S_k is linear in xi the moyal product ends at first order
    """
    w_new = f * gaussian_wigner + (1j/2) * poisson_bracket(f, gaussian_wigner, dx, dp, n_modes)
    return w_new

def poisson_bracket(f, g, dx, dp, n_modes):
    """
    Compute the Poisson bracket {f, g} over a 2N-dimensional phase-space grid.
 
    f and g are flat 1D arrays of length (n_q * n_p)^n_modes.
    We infer the per-axis grid size from the total number of points and
    reshape into an n_modes-pairs-of-axes grid before differentiating,
    then flatten back.
    """
    n_points = f.shape[0]
    n_axes = 2 * n_modes
    n_per_axis = round(n_points ** (1.0 / n_axes))
 
    if n_per_axis ** n_axes != n_points:
        raise ValueError(
            f"Cannot reshape {n_points} points into a {n_axes}-D grid "
            f"with equal side length. Got n_per_axis={n_per_axis}."
        )
 
    grid_shape = (n_per_axis,) * n_axes
 
    f_grid = f.reshape(grid_shape)
    g_grid = g.reshape(grid_shape)
 
    pb_grid = np.zeros(grid_shape, dtype=complex)
 
    for i in range(n_modes):
        x_axis = 2 * i
        p_axis = 2 * i + 1
 
        df_dx = np.gradient(f_grid, dx, axis=x_axis)
        df_dp = np.gradient(f_grid, dp, axis=p_axis)
 
        dg_dx = np.gradient(g_grid, dx, axis=x_axis)
        dg_dp = np.gradient(g_grid, dp, axis=p_axis)
 
        pb_grid += df_dx * dg_dp - df_dp * dg_dx
 
    return pb_grid.ravel()