import numpy as np

N_vals      = 1000
arr         = np.zeros(shape=(N_vals,3))
arr[:,0]    = np.linspace(-0.1,0.1,N_vals)

np.savetxt("scripts/1_funding_period/iv_curves/temperature/volt.csv", arr)