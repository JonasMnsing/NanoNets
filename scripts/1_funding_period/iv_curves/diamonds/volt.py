import numpy as np

N_vals          = 200
voltage_vals    = np.linspace(-0.1,0.1,N_vals)
arr             = np.zeros(shape=(N_vals*N_vals,9))

for i in range(N_vals):

    arr[i*N_vals:(i+1)*N_vals,0]    = voltage_vals
    arr[i*N_vals:(i+1)*N_vals,-1]   = np.repeat(voltage_vals[i],N_vals)
    
np.savetxt("scripts/1_funding_period/iv_curves/diamonds/volt.csv", arr)