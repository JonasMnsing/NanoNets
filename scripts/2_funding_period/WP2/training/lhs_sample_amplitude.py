from pyDOE import lhs
import numpy as np
import sys
import multiprocessing

from scipy import signal

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/home/j/j_mens07/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
sys.path.append("/home/jonasmensing/phd/NanoNets/src/")

# NanoNets Simulation Tool
import nanonets
import nanonets_utils

def run_sim(thread, x, params, rows, time_steps, topology, path, freq):

    thread_rows     = rows[thread]
    params_new      = params[thread_rows,:]

    for n, par in enumerate(params_new):
        N_voltages      = len(time_steps)
        voltages        = np.zeros((N_voltages, len(topology["e_pos"])+1))
        voltages[:,0]   = x
        for i, p in enumerate(par):
            voltages[:,i+1] = p*np.cos(freq*time_steps*1e8)

        # Run Training
        sim_class   = nanonets.simulation(topology_parameter=topology, folder=path, seed=0, add_to_path=f'_{thread}_{n}')
        sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=200)

path    = "scripts/2_funding_period/WP2/training/data/random_sample_amplitude/"
N_p     = 7

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

N_samples   = 500
N_procs     = 10
amplitude   = 0.02
freq        = 3.0
time_step   = 1e-10
N_periods   = 50
N_voltages  = int(N_periods*np.pi/(freq*1e8*time_step))
time_steps  = time_step*np.arange(N_voltages)
x_vals      = amplitude*np.cos(freq*time_steps*1e8)
N_controls  = len(topology_parameter["e_pos"])-2
p_range     = [[0.0,0.05] for _ in range(N_controls)]
lhs_sample  = lhs(N_controls, N_samples)
lhs_rescale = np.zeros_like(lhs_sample)

index   = [i for i in range(N_samples)]
rows    = [index[i::N_procs] for i in range(N_procs)]

for i in range(N_controls):
    lhs_rescale[:,i] = p_range[i][0] + lhs_sample[:, i] * (p_range[i][1] - p_range[i][0])

procs = []

for i in range(N_procs):
    
    process = multiprocessing.Process(target=run_sim, args=(i, x_vals, lhs_rescale, rows, time_steps, topology_parameter, path, freq))
    process.start()