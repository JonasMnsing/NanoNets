"""
Run KMC Code for const set of voltages with fixed parameters defined in "params.csv" @ folder.
Output Electrode @ last position in topology_parameter key "pos"
"""

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
def parallel_code(thread, voltages, time_steps, topology_parameter, res_info, eq_steps, folder, np_info, np_info2, T_val, save_th, stat_size):

    target_electrode    = len(topology_parameter["e_pos"]) - 1
    r_val               = int(np_info2["mean_radius"])
    
    for s in range(stat_size):

        sim_class = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter, folder=folder+f"data/r_{r_val}/r3/", res_info=res_info, np_info=np_info, np_info2=np_info2, add_to_path=f'_t{thread}_s{s}')
        sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, T_val=T_val, save_th=save_th)

if __name__ == '__main__':

    folder      = "scripts/2_funding_period/WP2/step_input/1I_1O_radius_dis/"
    voltages    = np.loadtxt(folder+'volt.csv')
    time_steps  = np.loadtxt(folder+'time.csv')
    stat_size   = 10

    N_processes, network_topology, topology_parameter, eq_steps, np_info, res_info, T_val, save_th = nanonets_utils.load_time_params(folder=folder)

    np_info2 = {
        "mean_radius"   : 5.0,
        "std_radius"    : 0.0,
        "np_index"      : np.random.choice(np.arange(49),24,replace=False)
    }

    procs = []
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, voltages, time_steps, topology_parameter, res_info, eq_steps, folder,
                                                                      np_info, np_info2, T_val, save_th, stat_size))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()

    np_info2 = {
        "mean_radius"   : 20.0,
        "std_radius"    : 0.0,
        "np_index"      : np.random.choice(np.arange(49),24,replace=False)
    }

    procs = []
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, voltages, time_steps, topology_parameter, res_info, eq_steps, folder,
                                                                      np_info, np_info2, T_val, save_th, stat_size))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()

    np_info2 = {
        "mean_radius"   : 40.0,
        "std_radius"    : 0.0,
        "np_index"      : np.random.choice(np.arange(49),24,replace=False)
    }

    procs = []
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, voltages, time_steps, topology_parameter, res_info, eq_steps, folder,
                                                                      np_info, np_info2, T_val, save_th, stat_size))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()
    