from optimization.optimize import optimize_stornati
import numpy as np

print("Still working on this...")


# Think of other way than the xi grid for better performance

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

    n_modes    = 2
    n_points   = 15
    limit      = 4.0
    L_max      = 8
    n_restarts = 3

    optimize_stornati(build_grid(n_modes, n_points, ))


if __name__ == "__main__":
    main()