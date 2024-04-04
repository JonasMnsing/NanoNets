import numpy as np

# Time Scale
step_size   = 1e-8
N_voltages  = 10000
time        = step_size*np.arange(N_voltages)

# Voltages
off_state   = 0.1
on_state    = 0.2
on_t1       = int(N_voltages/4)
on_t2       = int(3*N_voltages/4)
voltages    = np.zeros(shape=(N_voltages,3))

# Input Electrode
voltages[:,0]           = np.repeat(off_state, N_voltages)
voltages[on_t1:on_t2]   = on_state

np.savetxt("scripts/2_funding_period/WP2/step_input/volt.csv", voltages)
np.savetxt("scripts/2_funding_period/WP2/step_input/time.csv", time)