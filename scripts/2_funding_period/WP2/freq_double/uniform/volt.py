import numpy as np

# Time Scale
step_size   = 1e-10
amplitude   = 0.5
freq1       = 2e8
freq2       = 7e8
N_voltages  = 2000
time        = step_size*np.arange(N_voltages)

# Voltages
voltages    = np.zeros(shape=(N_voltages,4))

# Input Electrode
voltages[:,0] = amplitude*np.cos(freq1*time)
voltages[:,1] = amplitude*np.cos(freq2*time)

np.savetxt("scripts/2_funding_period/WP2/freq_double/uniform/volt.csv", voltages)
np.savetxt("scripts/2_funding_period/WP2/freq_double/uniform/time.csv", time)