import numpy as np
from optimization.LBFGSb import optimize_stornati

def main():
    print("| | |- Stornati Variational Optimization -| | |")

    n_modes = 2
    n_points = 15 # per axis; 4D grid → 15^4 = 50625 points (manageable)
    limit = 4.0

    print("Building phase space grid...")
    q_axis = np.linspace(-limit, limit, n_points)
    p_axis = np.linspace(-limit, limit, n_points)

    axes = [q_axis, p_axis] * n_modes
    mesh = np.meshgrid(*axes, indexing='ij')

    xi_grid = np.array([m.ravel() for m in mesh])
    print(f"Grid shape: {xi_grid.shape} (Dimensions: 2N x Points)")

    print(f"\nStarting L-BFGS-B optimization for {n_modes} mode(s)...")

    result, best_O, best_Z = optimize_stornati(xi_grid, n_modes)

    if result.success:
        print("\nOptimization Successful!")
        print(f"Minimum Energy Found: {result.fun:.6f}")
        print("\n--- Optimized Parameters ---")
        r_vals = np.diag(best_Z)[:n_modes]
        print(f"Best squeezing r = {r_vals}")
        print(f"Best squeezing in dB = {-20*np.log10(r_vals)}")
        print(f"Best O Matrix (2N×2N symplectic-orthogonal):\n{np.round(best_O, 4)}")
    else:
        print(f"\nOptimization Failed: {result.message}")
        print(f"Best energy reached: {result.fun:.6f}")

if __name__ == "__main__":
    main()