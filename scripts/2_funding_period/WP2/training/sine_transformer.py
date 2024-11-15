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
path                = "scripts/2_funding_period/WP2/training/data/sine_to_triangle/"
network_topology    = 'cubic'
learning_rate       = 0.01
batch_size          = 500
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
    "std_radius"    : 2.0,
    "np_distance"   : 1.0
}

# Voltage Values
amplitude   = 0.1
freq        = 3.5
time_step   = 1e-10
N_periods   = 50
N_voltages  = int(N_periods*np.pi/(freq*1e8*time_step))
time_steps  = time_step*np.arange(N_voltages)
x_vals      = amplitude*np.cos(freq*time_steps*1e8)
y_target    = amplitude*signal.sawtooth(freq*time_steps*1e8-np.pi, 0.5)

# Run Training
sim_class = nanonets.simulation(topology_parameter=topology_parameter, seed=0)
sim_class.train_time_series(x=x_vals, y=y_target, learning_rate=learning_rate, batch_size=batch_size, N_epochs=N_epochs,
                            adam=True, time_step=time_step, stat_size=stat_size, path=path)
