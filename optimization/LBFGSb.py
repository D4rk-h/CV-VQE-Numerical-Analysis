import numpy as np
from scipy.optimize import minimize
from model.math.polynomial import compute_variational_ansatz, unpack_params
from model.math.gaussian import n_unitary_params
from model.math.hamiltonian import H_BoseHubbard


def optimize_stornati(xi, n_modes, n_restarts=5, seed=42):
    """
    xi : (2N, n_points) — flattened phase-space grid from main.py.
 
    Runs L-BFGS-B from `n_restarts` random starting points and returns
    the best result. This is necessary because the energy landscape has
    local minima and flat directions at symmetric points (e.g. all-zeros).
    """
    H_symbol = H_BoseHubbard(xi.T, n_modes)
 
    q_unique = np.unique(np.round(xi[0], 10))
    dx = float(q_unique[1] - q_unique[0]) if len(q_unique) > 1 else 0.2
    dp = dx
    dV = dx ** (2 * n_modes)
 
    n_squeeze = n_modes
    n_angles  = n_unitary_params(n_modes)
    n_params  = n_squeeze + n_angles
 
    def objective(p):
        O_mat, Z_mat = unpack_params(p, n_modes)
        W = compute_variational_ansatz(xi, dx, dp, n_modes, O_mat, Z_mat)
        if not np.all(np.isfinite(W)):
            return 1e10
        energy = np.sum(H_symbol * W) * dV
        return float(np.real(energy)) if np.isfinite(energy) else 1e10
 
    rng = np.random.default_rng(seed)
    best_result = None
 
    for i in range(n_restarts):
        p0 = rng.normal(0, 0.5, n_params)

        result = minimize(
            objective,
            p0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-7}
        )
        print(f"  Restart {i+1}/{n_restarts}: E = {result.fun:.6f}"
              f"  ({'converged' if result.success else 'stopped'})")
 
        if best_result is None or result.fun < best_result.fun:
            best_result = result
 
    print(f"\nBest energy across {n_restarts} restarts: {best_result.fun:.6f}")
    best_O, best_Z = unpack_params(best_result.x, n_modes)
    return best_result, best_O, best_Z
 