import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
sys.path.append("/home/jonasmensing/phd/NanoNets/src/")

# NanoNets Simulation Tool
import nanonets
import nanonets_utils

# Parameter
N_epochs            = 100
transient_steps     = 100
stat_size           = 20
eq_steps            = 0
folder              = "scripts/2_funding_period/WP2/lorenz_system/"
save_folder         = "scripts/2_funding_period/WP2/lorenz_system/data/"
network_topology    = 'cubic'
learning_rate       = 0.00001
save_nth_epoch      = 1

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

N_voltages  = 1000
x_vals      = np.random.normal(loc=0.0, scale=0.05, size=N_voltages)
shift       = 1
y_target    = x_vals[:-shift]
x_vals      = x_vals[shift:]

sim_class = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter, np_info=np_info)
nanonets_utils.time_series_gradient_decent(sim_class=sim_class, x_vals=x_vals, y_target=y_target,
                                            learning_rate=learning_rate, N_epochs=N_epochs, print_info=True)
