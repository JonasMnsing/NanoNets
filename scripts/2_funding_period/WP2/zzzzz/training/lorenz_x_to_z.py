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
N_epochs            = 50
stat_size           = 500
eq_steps            = 0
path                = "scripts/2_funding_period/WP2/training/data/lorenz/"
network_topology    = 'cubic'
learning_rate       = 0.01
batch_size          = 1000
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
z_target    = np.loadtxt("scripts/2_funding_period/WP2/training/data/lorenz/z_vals.csv")
N_voltages  = len(x_vals)
time_step   = 1e-10
N_periods   = 50
time_steps  = time_step*np.arange(N_voltages)

# Run Training
sim_class = nanonets.simulation(topology_parameter=topology_parameter, seed=0)
# sim_class.train_time_series(x=x_vals, y=y_target, learning_rate=learning_rate, batch_size=batch_size, N_epochs=N_epochs,
#                             adam=True, time_step=time_step, stat_size=stat_size, path=path)

sim_class.train_time_series_by_frequency(x=x_vals, y=z_target, learning_rate=learning_rate, batch_size=batch_size, N_epochs=N_epochs,
                                        adam=True, time_step=time_step, stat_size=stat_size, path=path)
