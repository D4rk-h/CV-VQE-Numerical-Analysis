import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
import matplotlib.pyplot as plt
from optimization.LBFGSb import sweep_U


def plot_U_sweep(results, ED_results, U_values, L_values, sps_values,
                 save_path='U_sweep.png'):
    fig, ax = plt.subplots(figsize=(8, 6))

    colors    = ['tab:blue', 'tab:orange', 'tab:green']
    pa_labels = ['zero PA', 'one PA', 'two PA']
    markers   = ['o', 's', 'D']

    for sps, color in zip(sps_values, colors):
        ed_energies = [ED_results[sps][U] for U in U_values]
        ax.plot(U_values, ed_energies, color=color,
                label=f'ED_results sps={sps}', linewidth=1.5)

    for L, color, label, marker in zip(L_values, colors, pa_labels, markers):
        vqe_energies = [results[U].get(L, np.nan) for U in U_values]
        ax.scatter(U_values, vqe_energies, color=color, marker=marker,
                   label=f'CV_VQE_results {label}', s=40, zorder=5)

    ax.set_xlabel('U')
    ax.set_ylabel('$E_{GS}$')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


def print_best_result(summary, n_modes):
    """Prints the best result found across all L values."""
    best_L      = min(summary, key=lambda L: summary[L]['best_energy'])
    best        = summary[best_L]
    r_vals      = best['best_params'][:n_modes]

    print(f"\n{'='*50}")
    print(f"Best result at L = {best_L}")
    print(f"  Minimum energy : {best['best_energy']:.6f}")
    print(f"  Variance       : {best['variance_final']:.2e}")
    print(f"  Squeezing r    : {r_vals}")
    print(f"  Squeezing [dB] : {-20 * np.log10(np.abs(r_vals) + 1e-300)}")
    print(f"{'='*50}")


def build_grid(n_modes, n_points=15, limit=4.0):
    q_axis = np.linspace(-limit, limit, n_points)
    p_axis = np.linspace(-limit, limit, n_points)
    axes   = [q_axis, p_axis] * n_modes
    mesh   = np.meshgrid(*axes, indexing='ij')
    xi     = np.array([m.ravel() for m in mesh])
    print(f"Grid shape: {xi.shape}  (2N × Points)")
    return xi


def main():
    print("| | |- Stornati Variational Optimization -| | |")

    n_modes    = 4
    n_points   = 5
    limit      = 4.0
    L_max      = 8
    n_restarts = 3

    MODE = 'U_sweep'
    print("Building phase space grid...")
    xi_grid = build_grid(n_modes, n_points, limit)

    if MODE == 'U_sweep':
        U_values = np.linspace(0.05, 0.25, 5)

        print(f"\nRunning U-sweep over {len(U_values)} values, "
            f"L_max={L_max}, {n_restarts} restarts each...")
        results = sweep_U(
            U_values, xi_grid, n_modes,
            L_max=L_max,
            n_restarts=n_restarts,
            seed=42,
            t=1.0, mu=1.0,
        )

        ED_results = {10: {}, 20: {}, 30: {}}

        LATTICE_L = 4
        for sps in [10, 20, 30]:
            basis = boson_basis_1d(L=LATTICE_L, sps=sps)
            for U in U_values:
                t_val  = 1.0
                mu_val = 1.0
                static = [
                    ["+-", [[t_val, i, (i+1) % LATTICE_L] for i in range(LATTICE_L)]],
                    ["-+", [[t_val, i, (i+1) % LATTICE_L] for i in range(LATTICE_L)]],
                    ["n",  [[-mu_val, i]                   for i in range(LATTICE_L)]],
                    ["nn", [[U / 2,   i, i]                for i in range(LATTICE_L)]],
                ]
                no_checks = dict(check_herm=False, check_pcon=False, check_symm=False)
                H = hamiltonian(static, [], basis=basis, dtype=np.float64, **no_checks)
                E0 = H.eigsh(k=1, which='SA', return_eigenvectors=False)[0]
                ED_results[sps][U] = E0

        plt.figure()
        for sps, color in zip([10, 20, 30], ['C0', 'C1', 'C2']):
            x = np.array(sorted(ED_results[sps].keys()))
            y = np.array([ED_results[sps][u] for u in x])
            plt.plot(x, y, '-o', color=color, label=f'ED sps={sps}')

        plt.xlabel('U')
        plt.ylabel(r'$E_{GS}$')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plot_U_sweep(
            results, ED_results,
            U_values=U_values,
            L_values=[4, 6, 8],
            sps_values=[10, 20, 30],
        )

if __name__ == "__main__":
    main()