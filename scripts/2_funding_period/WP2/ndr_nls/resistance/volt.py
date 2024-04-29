import numpy as np
import sys
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets_utils

V_abs   = 0.05
G_abs   = 0.1
V_off   = 0.0
V_on    = 0.01

N_voltages      = 80000
N_electrodes    = 8

voltages        = nanonets_utils.logic_gate_config(low=-V_abs, high=V_abs, off_state=V_off, on_state=V_on, i1_col=1, i2_col=3, N_rows=N_voltages, N_cols=N_electrodes)
voltages[:,-1]  = 0
gates           = nanonets_utils.logic_gate_config_G(low=-G_abs, high=G_abs, N_rows=N_voltages)
voltages        = np.hstack((voltages,gates[np.newaxis].T))

np.savetxt("scripts/2_funding_period/WP2/ndr_nls/resistance/volt.csv", voltages, fmt='%1.5f')
