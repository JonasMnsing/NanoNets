import numpy as np

# Time Scale
step_size   = 1e-10
amplitude   = 0.2
freq        = 20e7
N_voltages  = 2000
time        = step_size*np.arange(N_voltages)

# Voltages
voltages    = np.zeros(shape=(N_voltages,4))

# Input Electrode
voltages[:,0] = amplitude*np.cos(freq*time)
voltages[:,1] = amplitude*np.cos(freq*time-np.pi/2)

np.savetxt("scripts/2_funding_period/WP2/freq_double/R_dis/volt.csv", voltages)
np.savetxt("scripts/2_funding_period/WP2/freq_double/R_dis/time.csv", time)