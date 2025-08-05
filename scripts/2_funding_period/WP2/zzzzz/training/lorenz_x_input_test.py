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
stat_size           = 100
eq_steps            = 0
path                = "scripts/2_funding_period/WP2/training/data/lorenz/test/"
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
x_vals      = np.loadtxt("scripts/2_funding_period/WP2/training/data/lorenz/x_vals.csv")
x_vals      = amplitude*(x_vals - np.mean(x_vals))/np.std(x_vals)
controls    = np.random.uniform(-amplitude, amplitude, 6)*10
N_voltages  = 1000

# voltages            = np.zeros(shape=(N_voltages,9))
# voltages[:,0]       = x_vals[:N_voltages]
# voltages[:,1:-2]    = np.tile(np.round(controls,4), (N_voltages,1))


time_step   = 1e-10
time_steps  = time_step*np.arange(N_voltages)

freqs               = [1.0,1.3,0.7,0.5,2.2,1.1]
voltages            = np.zeros(shape=(N_voltages,9))
for i, f in enumerate(freqs):
    voltages[:,i+1] = amplitude*np.cos(f*time_steps*1e8)
voltages[:,0]       = x_vals[:N_voltages]

print(voltages)

# Run Training
sim_class   = nanonets.simulation(topology_parameter=topology_parameter, seed=0, folder=path)
sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=stat_size, output_potential=True)