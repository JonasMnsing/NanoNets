import numpy as np
import sys
import time

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets
import nanonets_utils
import multiprocessing

# Simulation Function
def parallel_code(thread, voltages, time_steps, topology_parameter, res_info, eq_steps, folder, np_info, np_info2, T_val, save_th, add_to_path, stat_size, seed):

    target_electrode = len(topology_parameter["e_pos"]) - 1
    
    for s in range(stat_size):

        sim_class = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter, folder=folder, res_info=res_info, np_info=np_info, np_info2=np_info2, add_to_path=add_to_path+f'_t{thread}_s{s}', seed=seed)
        sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, T_val=T_val, save_th=save_th)

if __name__ == '__main__':

    folder      = "scripts/2_funding_period/WP2/spatial_correlation/radius_corr/"
    voltages    = np.loadtxt(folder+'volt.csv')
    time_steps  = np.loadtxt(folder+'time.csv')
    stat_size   = 5
    seed        = int(time.monotonic_ns()/10000)

    # Set Seed
    np.random.seed(seed)

    N_processes, network_topology, topology_parameter, eq_steps, np_info, res_info, T_val, save_th, add_to_path = nanonets_utils.load_time_params(folder=folder)

    np_info2 = {
            "mean_radius"   : 20.0,
            "std_radius"    : 0.0,
            "np_index"      : [16,17,18,23,24,25,30,31,32]
        }

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, voltages, time_steps, topology_parameter, res_info, eq_steps, folder,
                                                                      np_info, np_info2, T_val, save_th, add_to_path, stat_size, seed))
        process.start()