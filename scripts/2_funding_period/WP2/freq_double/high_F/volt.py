import numpy as np

# Time Scale
step_size   = 1e-11
amplitude   = 0.2
freq1       = 2e9
freq2       = 7e9
N_voltages  = 4000
time        = step_size*np.arange(N_voltages)

# Voltages
voltages    = np.zeros(shape=(N_voltages,4))

# Input Electrode
voltages[:,0] = amplitude*np.cos(freq1*time)
voltages[:,1] = amplitude*np.cos(freq2*time)

np.savetxt("scripts/2_funding_period/WP2/freq_double/high_F/volt.csv", voltages)
np.savetxt("scripts/2_funding_period/WP2/freq_double/high_F/time.csv", time)