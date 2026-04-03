from model.math.prep import build_O, build_Z, build_cov, build_displacement, sample_xi
from simulation.simulate import run_simulation
from plotting.plot_wigner import plot_wigner_plotly

N         = 2
n_samples = 200_000
t, U, mu  = 0.5, 1, 0.1
j_mode    = 0

O       = build_O(N)                        
Z       = build_Z(0.5, N)                   
V       = build_cov(O, Z)                   
xi_mean = build_displacement(N)             

samples = sample_xi(xi_mean, V, n_samples)  

for L_val in [1, 2, 4, 8]:
    P, energy = run_simulation(samples, V, L_val, N, j=j_mode)
    plot_wigner_plotly(samples, P, N, L_val, j=j_mode)