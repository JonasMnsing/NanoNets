import numpy as np
import sys

from scipy import signal

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/home/j/j_mens07/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
sys.path.append("/home/jonasmensing/phd/NanoNets/src/")

# NanoNets Simulation Tool
import nanonets
import nanonets_utils

# Hyper Parameter
stat_size           = 500
path                = "scripts/2_funding_period/WP2/training/data/sine_example/"
network_topology    = 'cubic'
N_p                 = 7

# Network Topology
topology_parameter  = {
    "Nx"                : N_p,
    "Ny"                : N_p,
    "Nz"                : 1,
    "e_pos"             : [[0,0,0],[(N_p-1)//2,0,0],[0,(N_p-1)//2,0],[N_p-1,0,0],[0,N_p-1,0],[N_p-1,(N_p-1)//2,0],[(N_p-1)//2,N_p-1,0],[N_p-1,N_p-1,0]],
    "electrode_type"    : ['constant','constant','constant','constant','constant','constant','constant','floating']
}
np_info = {
    "eps_r"         : 2.6, 
    "eps_s"         : 3.9,
    "mean_radius"   : 10.0,
    "std_radius"    : 0.0,
    "np_distance"   : 1.0
}


# Voltage Values
amplitude   = 0.1
freqs       = [2.0, 1.0, 3.0, 4.0, 2.2, 1.6, 4.5]
time_step   = 1e-10
N_periods   = 50
N_voltages  = int(N_periods*np.pi/(freqs[0]*1e8*time_step))
voltages    = np.zeros(shape=(N_voltages,len(topology_parameter["e_pos"])+1))
time_steps  = time_step*np.arange(N_voltages)

for i, f in enumerate(freqs):
    voltages[:,i] = amplitude*np.cos(f*time_steps*1e8)

sim_class   = nanonets.simulation(topology_parameter=topology_parameter, folder=path, add_to_path='')
sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=stat_size)