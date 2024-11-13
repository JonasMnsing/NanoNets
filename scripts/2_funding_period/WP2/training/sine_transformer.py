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

# Parameter
N_epochs            = 100
stat_size           = 10
eq_steps            = 0
path                = "scripts/2_funding_period/WP2/training/data/sine_to_triangle/"
network_topology    = 'cubic'
learning_rate       = 0.001

# Network Topology
topology_parameter  = {
    "Nx"                : 7,
    "Ny"                : 7,
    "Nz"                : 1,
    "e_pos"             : [[0,0,0],[3,0,0],[0,3,0],[6,0,0],[0,6,0],[6,3,0],[3,6,0],[6,6,0]],
    "electrode_type"    : ['constant','constant','constant','constant','constant','constant','constant','floating']
}
np_info = {
    "eps_r"         : 2.6, 
    "eps_s"         : 3.9,
    "mean_radius"   : 10.0,
    "std_radius"    : 0.0,
    "np_distance"   : 1.0
}

# N_voltages  = 2000
amplitude   = 0.1
freq        = 1.0
time_step   = 1e-10
N_voltages  = int(2*np.pi/(freq*1e8*time_step))
time_steps  = time_step*np.arange(N_voltages)
x_vals      = amplitude*np.cos(freq*time_steps*1e8)
y_target    = amplitude*signal.sawtooth(freq*time_steps*1e8, 0.5)

nanonets_utils.time_series_gradient_decent(x_vals=x_vals, y_target=y_target, time_step=time_step, learning_rate=learning_rate,
                                           N_epochs=N_epochs, network_topology='cubic', topology_parameter=topology_parameter,
                                           print_nth_epoch=1, stat_size=stat_size, adam=False, path=path)
