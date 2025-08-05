import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
sys.path.append("/home/j/j_mens07/phd/NanoNets/src/")

import nanonets
import nanonets_utils

# Network Paramter
N_p                 = 9
folder              = "scripts/2_funding_period/WP2/"
topology_parameter  = {
    "Nx"                : N_p,
    "Ny"                : N_p,
    "Nz"                : 1,
    "e_pos"             : [[(N_p-1)//2,0,0],[0,0,0],[N_p-1,0,0],[0,(N_p-1)//2,0],
                           [N_p-1,(N_p-1)//2,0],[0,N_p-1,0],[N_p-1,N_p-1,0],[(N_p-1)//2,N_p-1,0]],
    "electrode_type"    : ['constant','constant','constant','constant','constant','constant','constant','floating']
}
target_electrode    = len(topology_parameter["e_pos"])-1
np_info = {
    "eps_r"         : 2.6, 
    "eps_s"         : 3.9,
    "mean_radius"   : 10.0,
    "std_radius"    : 0.0,
    "np_distance"   : 1.0
}

# Voltage Paramter
N_voltages              = 10000
amplitudes              = [0.01,0.0,-0.02,0.0,0.01,0.05,0.0,0.0]
frequencies             = [4.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0]
phase                   = 0.0
offset                  = 0.0
time_steps, voltages    = nanonets_utils.sinusoidal_voltages(N_samples=N_voltages, topology_parameter=topology_parameter,
                                                             amplitudes=amplitudes, frequencies=frequencies, phase=phase, offset=offset)

sim_class = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, np_info=np_info)
sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode)