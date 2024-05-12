import numpy as np

# Time Scale
step_size   = 1e-10
amplitude   = 0.05
freq        = 80e7
N_voltages  = 2000
time        = step_size*np.arange(N_voltages)

# Voltages
voltages    = np.zeros(shape=(N_voltages,3))

# Input Electrode
voltages[:,0] = amplitude*np.cos(freq*time) + 0.15

np.savetxt("scripts/2_funding_period/WP2/cos_input/1I_1O_R_corr/volt.csv", voltages)
np.savetxt("scripts/2_funding_period/WP2/cos_input/1I_1O_R_corr/time.csv", time)