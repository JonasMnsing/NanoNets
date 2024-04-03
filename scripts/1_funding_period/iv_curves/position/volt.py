import numpy as np

N_vals                      = 1000
arr                         = np.zeros(shape=(4*N_vals,9))
arr[:N_vals,0]              = np.linspace(-0.1,0.1,N_vals)
arr[N_vals:2*N_vals,1]      = np.linspace(-0.1,0.1,N_vals)
arr[2*N_vals:3*N_vals,2]    = np.linspace(-0.1,0.1,N_vals)
arr[3*N_vals:,3]            = np.linspace(-0.1,0.1,N_vals)

np.savetxt("scripts/1_funding_period/iv_curves/position/volt.csv", arr)