import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/jonas/phd/NanoNets/src/")
# sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
import model
import multiprocessing

if __name__ == '__main__':

    N_processes         = 10
    N_electrodes        = 8
    N_voltages          = 500
    max_time            = 2e-4
    frequency           = 0.5e5
    time_steps          = np.linspace(0,max_time,N_voltages)
    amplitude           = 0.05

    def parallel_code(thread, N_voltages, frequency, time_steps, amplitude, N_electrodes):

        for inner_stat in range(10): 
            N                           = 7
            voltages_values             = amplitude*np.cos(2*np.pi*frequency*time_steps)
            topology_parameter          = {}
            topology_parameter["Nx"]    = N
            topology_parameter["Ny"]    = N
            topology_parameter["Nz"]    = 1
            topology_parameter["e_pos"] = [[0,0,0], [int((N-1)/2),0,0], [N-1,0,0], [0,int((N-1)/2),0], [0,N-1,0], [N-1,int((N)/2),0], [int((N)/2),(N-1),0], [N-1,N-1,0]]

            target_electrode    = len(topology_parameter["e_pos"]) - 1
            folder              = "/home/jonas/phd/NanoNets/test_runs/time_run/"#"/scratch/tmp/j_mens07/data/system_size/"
            voltages            = pd.DataFrame(np.zeros((N_voltages, N_electrodes+1)))
            voltages.iloc[:,0]  = voltages_values

            sim_class   = model.simulation(voltages.values)
            sim_class.init_cubic(folder, topology_parameter, add_to_path=f"_{thread}_{inner_stat}")
            sim_class.run_var_voltages(target_electrode, time_steps, save_th=1)
            # model.time_simulation(target_electrode, time_steps, topology_parameter, voltages.values, folder, add_to_path=f"_{thread}_f{f_mult}",
            #                             save_th=1, T_val=0.0)
    
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, N_voltages, frequency, time_steps, amplitude, N_electrodes))
        process.start()