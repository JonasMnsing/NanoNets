import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/jonas/phd/NanoNets/src/")
# sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
import simulation
import multiprocessing

if __name__ == '__main__':

    N_processes         = 10
    N_electrodes        = 2
    N_voltages          = 200
    max_time            = 1e-4
    frequency           = 1e5
    time_steps          = np.linspace(0,max_time,N_voltages)
    amplitude           = 0.05

    def parallel_code(thread, N_voltages, frequency, time_steps, amplitude, N_electrodes):

        for f_mult in [0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]:

            voltages_values             = amplitude*np.cos(f_mult*frequency*time_steps)
            topology_parameter          = {}
            topology_parameter["Nx"]    = 2
            topology_parameter["Ny"]    = 2
            topology_parameter["Nz"]    = 1
            topology_parameter["e_pos"] = [[0,0,0], [1,1,0]]

            target_electrode    = len(topology_parameter["e_pos"]) - 1
            folder              = "/home/jonas/phd/NanoNets/test_runs/time_"#"/scratch/tmp/j_mens07/data/system_size/"
            voltages            = pd.DataFrame(np.zeros((N_voltages,N_electrodes+1)))
            voltages.iloc[:,0]  = voltages_values

            simulation.time_simulation(target_electrode, time_steps, topology_parameter, voltages.values, folder, add_to_path=f"_{thread}_f{f_mult}",
                                        save_th=1, T_val=0.0)
    
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, N_voltages, frequency, time_steps, amplitude, N_electrodes))
        process.start()