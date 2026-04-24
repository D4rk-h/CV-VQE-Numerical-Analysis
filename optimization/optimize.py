from scipy.optimize import minimize
from scipy.linalg import expm
import numpy as np
from model.math.hamiltonian import h_bose_hubbard
from model.math.ansatz import compute_variational_ansatz_layered


def optimize_stornati(xi, n_modes=2, L_max=8, n_restarts=2, seed=42, U=1.0, t=1.0, mu=1.0):
    hamiltonian = h_bose_hubbard(xi, n_modes, t, U, mu)
    
    q_unique = np.unique(np.round(xi[0], 10))
    dx = float(q_unique[1] - q_unique[0]) if len(q_unique) > 1 else 0.2
    dp = dx
    dV = dx ** (2 * n_modes)
 
    n_params = n_modes + n_modes**2
 
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
            energy = np.sum(hamiltonian * W) * dV
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
                float(np.real(np.sum(hamiltonian * W) * dV))
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

def unpack_params(params: np.ndarray, n_modes: int):
    """
    Flat optimizer params → (O, Z).

    Layout  (length = n_modes + n_modes²):
      params[:n_modes]  — log-squeezing; r_i = exp(params_i)
      params[n_modes:]  — N² params for U(N) via skew-Hermitian expm
    """
    N = n_modes
    r = np.exp(np.clip(params[:N], -5, 5))
    Z = np.diag(np.concatenate([r, 1.0 / r]))

    S = _skew_hermitian_to_unitary(params[N:], N)
    U = expm(S)
    O = np.block([[np.real(U), -np.imag(U)],
                  [np.imag(U),  np.real(U)]])
    return O, Z

def _skew_hermitian_to_unitary(s: np.ndarray, N: int) -> np.ndarray:
    """Build skew-Hermitian S from flat params, ready for expm(S) → U(N)."""
    n_off  = N * (N - 1) // 2
    s_re   = s[:n_off]
    s_im   = s[n_off:2 * n_off]
    s_diag = s[2 * n_off:]

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