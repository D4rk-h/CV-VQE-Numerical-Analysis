import numpy as np
from scipy.optimize import minimize
from model.math.polynomial import compute_variational_ansatz_layered
from model.math.gaussian import n_unitary_params, unpack_params
from model.math.hamiltonian import H_BoseHubbard


def optimize_stornati_with_variance(xi, n_modes, L_max=None, n_restarts=2, seed=42, U=1.0, t=1.0, mu=1.0):
    if L_max is None:
        L_max = n_modes
 
    H_symbol = H_BoseHubbard(xi.T, n_modes, U, t, mu)
 
    q_unique = np.unique(np.round(xi[0], 10))
    dx = float(q_unique[1] - q_unique[0]) if len(q_unique) > 1 else 0.2
    dp = dx
    dV = dx ** (2 * n_modes)
 
    n_params = n_modes + n_unitary_params(n_modes)
 
    rng = np.random.default_rng(seed)
    summary = {}
 
    for L in range(2, L_max + 1, 2):
        print(f"\n=== L = {L} ===")
 
        def objective(p, _L=L):
            O_mat, Z_mat = unpack_params(p, n_modes)
            W_list = compute_variational_ansatz_layered(
                xi, dx, dp, n_modes, O_mat, Z_mat, _L
            )
            W = W_list[-1]
            if not np.all(np.isfinite(W)):
                return 1e10
            energy = np.sum(H_symbol * W) * dV
            return float(np.real(energy)) if np.isfinite(energy) else 1e10
 
        best_result = None
        best_energies_per_layer = None
        all_restart_energies_per_layer = []
 
        for i in range(n_restarts):
            p0 = rng.normal(0, 0.5, n_params)
            result = minimize(
                objective, p0, method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-7}
            )
 
            O_mat, Z_mat = unpack_params(result.x, n_modes)
            W_list = compute_variational_ansatz_layered(
                xi, dx, dp, n_modes, O_mat, Z_mat, L
            )
            energies_per_layer = [
                float(np.real(np.sum(H_symbol * W) * dV))
                for W in W_list
            ]
            all_restart_energies_per_layer.append(energies_per_layer)
 
            print(f"  Restart {i+1}/{n_restarts}: "
                  f"E(L={L}) = {result.fun:.6f}  "
                  f"({'converged' if result.success else 'stopped'})")
 
            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_energies_per_layer = energies_per_layer
 
        arr = np.array(all_restart_energies_per_layer)
        variance_per_layer = list(np.var(arr, axis=0))
        variance_final     = float(np.var(arr[:, -1]))
 
        summary[L] = {
            'best_energy'                      : best_result.fun,
            'best_params'                      : best_result.x,
            'energies_per_layer'               : best_energies_per_layer,
            'variance_final'                   : variance_final,
            'variance_per_layer'               : variance_per_layer,
            'all_restart_energies_per_layer'   : arr.tolist(),
        }
 
        print(f"  Best E = {best_result.fun:.6f}  |  "
              f"Variance (final layer) = {variance_final:.2e}")
 
    return summary


def sweep_U(U_values, xi, n_modes, L_max=None, n_restarts=2, seed=42, t=1.0, mu=1.0):
    results = {}
    for U in U_values:
        print(f"\n{'='*50}\nU = {U:.4f}")
        summary = optimize_stornati_with_variance(
            xi, n_modes,
            L_max=L_max,
            n_restarts=n_restarts,
            seed=seed,
            U=U, t=t, mu=mu,
        )
        results[U] = {L: summary[L]['best_energy'] for L in summary}
    return results