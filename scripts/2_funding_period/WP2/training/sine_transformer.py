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
stat_size           = 100
eq_steps            = 0
path                = "scripts/2_funding_period/WP2/training/data/sine_to_triangle/"
network_topology    = 'cubic'
learning_rate       = 0.01
epsilon             = 0.1
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
freq        = 2.0
time_step   = 1e-10
N_periods   = 20
N_voltages  = int(N_periods*np.pi/(freq*1e8*time_step))
batch_size  = N_voltages
time_steps  = time_step*np.arange(N_voltages)
x_vals      = amplitude*np.cos(freq*time_steps*1e8)
y_target    = amplitude*signal.square(freq*time_steps*1e8-3*np.pi/2)

# y_target    = amplitude*signal.sawtooth(freq*time_steps*1e8-np.pi, 0.5)

# Run Training
optim_class = nanonets.Optimizer("","freq",topology_parameter=topology_parameter, seed=0)
optim_class.gradient_decent(x=x_vals, y=y_target, N_epochs=N_epochs, learning_rate=learning_rate,
                            batch_size=batch_size, adam=True, time_step=time_step, epsilon=epsilon)
