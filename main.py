from model.math.gaussian import build_O, build_Z, build_cov, build_displacement, sample_xi
from simulation.simulate import run_simulation

N         = 2
n_samples = 200_000
t, U, mu  = 1, 1, 1
j_mode    = 0

O       = build_O(N)                        
Z       = build_Z(0.5, N)                   
V       = build_cov(O, Z)                   
xi_mean = build_displacement(N)             

samples = sample_xi(xi_mean, V, n_samples)  

Ls = [1, 2, 4, 8]
for L_val in Ls:
    P, energy = run_simulation(samples, V, L_val, N, j=j_mode)