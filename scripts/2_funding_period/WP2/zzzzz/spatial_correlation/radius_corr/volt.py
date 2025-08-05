import numpy as np

# Time Scale
step_size   = 1e-10
N_voltages  = 1000
time        = step_size*np.arange(N_voltages)

# Voltages
off_state   = 0.1
on_state    = 0.2
on_t1       = 100
on_t2       = 300
voltages    = np.zeros(shape=(N_voltages,3))

# Input Electrode
voltages[:,0]           = np.repeat(off_state, N_voltages)
voltages[on_t1:on_t2,0] = on_state

np.savetxt("scripts/2_funding_period/WP2/spatial_correlation/radius_corr/volt.csv", voltages)
np.savetxt("scripts/2_funding_period/WP2/spatial_correlation/radius_corr/time.csv", time)