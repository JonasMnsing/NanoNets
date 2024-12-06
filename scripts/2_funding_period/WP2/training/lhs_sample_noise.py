from pyDOE import lhs
import numpy as np
import sys
import multiprocessing

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/home/j/j_mens07/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
sys.path.append("/home/jonasmensing/phd/NanoNets/src/")

# NanoNets Simulation Tool
import nanonets

def return_lhs_sample(pmin, pmax, N_controls, N_samples):

    p_range     = [[pmin,pmax] for _ in range(N_controls)]
    lhs_sample  = lhs(N_controls, N_samples)
    lhs_rescale = np.zeros_like(lhs_sample)

    for i in range(N_controls):
        lhs_rescale[:,i] = p_range[i][0] + lhs_sample[:, i] * (p_range[i][1] - p_range[i][0])

    return lhs_rescale

def run_sim(thread, x, params, rows, time_steps, topology, path, sim_type, amplitude, freq):

    thread_rows     = rows[thread]
    params_new      = params[thread_rows,:]

    if sim_type == 'offset':

        for n, par in enumerate(params_new):
            N_voltages      = len(time_steps)
            voltages        = np.zeros((N_voltages, len(topology["e_pos"])+1))
            voltages[:,0]   = x
            for i, p in enumerate(par):
                voltages[:,i+1] = amplitude*np.cos(freq*time_steps*1e8) + p

            sim_class   = nanonets.simulation(topology_parameter=topology, folder=path, seed=0, add_to_path=f'_{thread}_{n}')
            sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=200)

    elif sim_type == 'amplitude':

        for n, par in enumerate(params_new):
            N_voltages      = len(time_steps)
            voltages        = np.zeros((N_voltages, len(topology["e_pos"])+1))
            voltages[:,0]   = x
            for i, p in enumerate(par):
                voltages[:,i+1] = p*np.cos(freq*time_steps*1e8)
            
            sim_class   = nanonets.simulation(topology_parameter=topology, folder=path, seed=0, add_to_path=f'_{thread}_{n}')
            sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=200)


    elif sim_type == 'frequency':

        for n, par in enumerate(params_new):
            N_voltages      = len(time_steps)
            voltages        = np.zeros((N_voltages, len(topology["e_pos"])+1))
            voltages[:,0]   = x
            for i, p in enumerate(par):
                voltages[:,i+1] = amplitude*np.cos(p*time_steps*1e8)

            sim_class   = nanonets.simulation(topology_parameter=topology, folder=path, seed=0, add_to_path=f'_{thread}_{n}')
            sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=200)

    elif sim_type == 'phase':

        for n, par in enumerate(params_new):
            N_voltages      = len(time_steps)
            voltages        = np.zeros((N_voltages, len(topology["e_pos"])+1))
            voltages[:,0]   = x
            for i, p in enumerate(par):
                voltages[:,i+1] = amplitude*np.cos(freq*time_steps*1e8 - p)

            sim_class   = nanonets.simulation(topology_parameter=topology, folder=path, seed=0, add_to_path=f'_{thread}_{n}')
            sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=200)

    else:

        for n, par in enumerate(params_new):
            N_voltages      = len(time_steps)
            voltages        = np.zeros((N_voltages, len(topology["e_pos"])+1))
            voltages[:,0]   = x
            for i, p in enumerate(par):
                voltages[:,i+1] = p

            sim_class   = nanonets.simulation(topology_parameter=topology, folder=path, seed=0, add_to_path=f'_{thread}_{n}')
            sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=200)


N_p = 7

# Network Topology
topology_parameter  = {
    "Nx"                : N_p,
    "Ny"                : N_p,
    "Nz"                : 1,
    "e_pos"             : [[(N_p-1)//2,0,0],[0,0,0],[N_p-1,0,0],
                           [0,(N_p-1)//2,0],[N_p-1,(N_p-1)//2,0],[0,N_p-1,0],
                           [N_p-1,N_p-1,0],[(N_p-1)//2,N_p-1,0]],
    "electrode_type"    : ['constant','constant','constant','constant','constant','constant','constant','floating']
}
np_info = {
    "eps_r"         : 2.6, 
    "eps_s"         : 3.9,
    "mean_radius"   : 10.0,
    "std_radius"    : 0.0,
    "np_distance"   : 1.0
}

# Parameter
N_samples   = 500
N_procs     = 10
amplitude   = 0.01
freq        = 3.5
time_step   = 1e-10
N_voltages  = 5000
time_steps  = time_step*np.arange(N_voltages)
x_vals      = np.random.uniform(-amplitude,amplitude,N_voltages)
N_controls  = len(topology_parameter["e_pos"])-2
index       = [i for i in range(N_samples)]
rows        = [index[i::N_procs] for i in range(N_procs)]

# Const
sim_type    = 'const'
path        = "scripts/2_funding_period/WP2/training/data/lhs_sample_noise/const/"
sample      = return_lhs_sample(-0.05, 0.05, N_controls, N_samples)
procs       = []
for i in range(N_procs):
    
    process = multiprocessing.Process(target=run_sim, args=(i, x_vals, sample, rows, time_steps,
                                                            topology_parameter, path, sim_type, amplitude, freq))
    process.start()
    procs.append(process)
for p in procs:
    p.join()

# # Offset
# sim_type    = 'offset'
# path        = "scripts/2_funding_period/WP2/training/data/lhs_sample_noise/offset/"
# sample      = return_lhs_sample(-0.05, 0.05, N_controls, N_samples)
# procs       = []
# for i in range(N_procs):
    
#     process = multiprocessing.Process(target=run_sim, args=(i, x_vals, sample, rows, time_steps,
#                                                             topology_parameter, path, sim_type, amplitude, freq))
#     process.start()
#     procs.append(process)
# for p in procs:
#     p.join()

# # Amplitude
# sim_type    = 'amplitude'
# path        = "scripts/2_funding_period/WP2/training/data/lhs_sample_noise/amplitude/"
# sample      = return_lhs_sample(0.0, 0.05, N_controls, N_samples)
# procs       = []
# for i in range(N_procs):
    
#     process = multiprocessing.Process(target=run_sim, args=(i, x_vals, sample, rows, time_steps,
#                                                             topology_parameter, path, sim_type, amplitude, freq))
#     process.start()
#     procs.append(process)
# for p in procs:
#     p.join()

# # Frequency
# sim_type    = 'frequency'
# path        = "scripts/2_funding_period/WP2/training/data/lhs_sample_noise/frequency/"
# sample      = return_lhs_sample(0.0, 6.0, N_controls, N_samples)
# procs       = []
# for i in range(N_procs):
    
#     process = multiprocessing.Process(target=run_sim, args=(i, x_vals, sample, rows, time_steps,
#                                                             topology_parameter, path, sim_type, amplitude, freq))
#     process.start()
#     procs.append(process)
# for p in procs:
#     p.join()


# # Phase
# sim_type    = 'phase'
# path        = "scripts/2_funding_period/WP2/training/data/lhs_sample_noise/phase/"
# sample      = return_lhs_sample(0.0, 2*np.pi, N_controls, N_samples)
# procs       = []
# for i in range(N_procs):
    
#     process = multiprocessing.Process(target=run_sim, args=(i, x_vals, sample, rows, time_steps,
#                                                             topology_parameter, path, sim_type, amplitude, freq))
#     process.start()
#     procs.append(process)
# for p in procs:
#     p.join()