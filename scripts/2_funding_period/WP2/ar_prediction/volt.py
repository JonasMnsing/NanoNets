import numpy as np

def generate_ar_process(p, phi, length):

    # Time Series
    series  = np.random.normal(0, 1, length)
    errors  = np.random.normal(0, 1, length)
    phi     = np.array(phi)

    for i in range(p,length):

        series[i] = np.sum(series[i-p:i] * phi) + errors[i]
    
    return series

# Time Scale
step_size   = 1e-10
amplitude   = 0.2
freq        = 10e7
N_voltages  = 1000
time        = step_size*np.arange(N_voltages)
ts          = generate_ar_process(1, [0.9], N_voltages)
ts          = 0.1*(ts - np.min(ts))/(np.max(ts)-np.min(ts)) + 0.1

# Voltages
voltages    = np.zeros(shape=(N_voltages,3))

# Input Electrode
voltages[:,0] = ts

np.savetxt("scripts/2_funding_period/WP2/ar_prediction/volt.csv", voltages)
np.savetxt("scripts/2_funding_period/WP2/ar_prediction/time.csv", time)