import numpy as np
import pandas as pd
import sys
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
import nanonets
import multiprocessing

if __name__ == '__main__':

    N_processes         = 10
    N_electrodes        = 2
    N_voltages          = 10000
    step_size           = 1e-8
    time_steps          = np.cumsum(np.repeat(step_size, N_voltages))
    amplitude           = 0.2

    def parallel_code(thread, N_voltages, time_steps, amplitude, N_electrodes):

        np.random.seed(thread)

        for inner_stat in range(5):

            min_val                     = -amplitude
            max_val                     = amplitude
            data                        = np.cumsum(np.random.normal(0,1,N_voltages))
            data_s                      = ((data - np.min(data))/(np.max(data)-np.min(data)))*(max_val - min_val) + min_val  
            voltages_values             = data_s
            topology_parameter          = {}
            topology_parameter["Np"]    = 50
            topology_parameter["Nj"]    = 4
            # topology_parameter["e_pos"] = [[-1.5,-1.5],[0,-1.5],[1.5,-1.5],[-1.5,0],[-1.5,1.5],[1.5,0],[0,1.5],[1.5,1.5]]
            topology_parameter["e_pos"] = [[-1.5,-1.5],[1.5,1.5]]

            target_electrode    = len(topology_parameter["e_pos"]) - 1
            # folder              = "/mnt/c/Users/jonas/Desktop/phd/NanoNets/test_runs/"
            # folder              = "/home/jonas/phd/NanoNets/test_runs/"
            folder              = "/mnt/c/Users/jonas/Desktop/phd/data/dis_topo/"
            voltages            = pd.DataFrame(np.zeros((N_voltages, N_electrodes+1)))
            voltages.iloc[:,0]  = voltages_values

            sim_class   = nanonets.simulation(voltages.values)
            sim_class.init_random(folder, topology_parameter, add_to_path=f"_{thread}_{inner_stat}")
            sim_class.run_var_voltages(target_electrode, time_steps, save_th=1, T_val=0.0)
    
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, N_voltages, time_steps, amplitude, N_electrodes))
        process.start()