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
def parallel_code(thread, voltages, time_steps, topology_parameter, res_info, res_info2, eq_steps, folder, np_info, T_val, save_th, stat_size, seed):

    target_electrode    = len(topology_parameter["e_pos"]) - 1
    R_val               = res_info2["R"]
    
    for s in range(stat_size):

        sim_class = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter, folder=folder+f"data/R_{R_val}/r{seed}/", res_info=res_info, res_info2=res_info2, np_info=np_info, add_to_path=f'_t{thread}_s{s}')
        sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, T_val=T_val, save_th=save_th)

if __name__ == '__main__':

    folder      = "scripts/2_funding_period/WP2/step_input/1I_1O_R_dis/"
    voltages    = np.loadtxt(folder+'volt.csv')
    time_steps  = np.loadtxt(folder+'time.csv')
    stat_size   = 10

    for seed in range(1,4):

        rs = np.random.RandomState(seed)

        N_processes, network_topology, topology_parameter, eq_steps, np_info, res_info, T_val, save_th = nanonets_utils.load_time_params(folder=folder)

        res_info2   = {
            "R" : 50,
            "np_index" : rs.choice(np.arange(1,48), 9, replace=False)
        }

        procs = []
        for i in range(N_processes):

            process = multiprocessing.Process(target=parallel_code, args=(i, voltages, time_steps, topology_parameter, res_info, res_info2, eq_steps, folder,
                                                                        np_info, T_val, save_th, stat_size, seed))
            process.start()
            procs.append(process)
        
        for p in procs:
            p.join()

        res_info2["R"] = 100

        procs = []
        for i in range(N_processes):

            process = multiprocessing.Process(target=parallel_code, args=(i, voltages, time_steps, topology_parameter, res_info, res_info2, eq_steps, folder,
                                                                        np_info, T_val, save_th, stat_size, seed))
            process.start()
            procs.append(process)
        
        for p in procs:
            p.join()

        res_info2["R"] = 200

        procs = []
        for i in range(N_processes):

            process = multiprocessing.Process(target=parallel_code, args=(i, voltages, time_steps, topology_parameter, res_info, res_info2, eq_steps, folder,
                                                                        np_info, T_val, save_th, stat_size, seed))
            process.start()
            procs.append(process)
        
        for p in procs:
            p.join()

        res_info2["R"] = 400

        procs = []
        for i in range(N_processes):

            process = multiprocessing.Process(target=parallel_code, args=(i, voltages, time_steps, topology_parameter, res_info, res_info2, eq_steps, folder,
                                                                        np_info, T_val, save_th, stat_size, seed))
            process.start()
            procs.append(process)
        
        for p in procs:
            p.join()