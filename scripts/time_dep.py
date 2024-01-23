import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
import nanonets
import multiprocessing

if __name__ == '__main__':

    N_processes         = 10
    N_electrodes        = 2
    N_voltages          = 10000
    # frequency           = 2e5
    step_size           = 1e-8
    time_steps          = np.cumsum(np.repeat(step_size, N_voltages))
    amplitude           = 0.2

    def parallel_code(thread, N_voltages, time_steps, amplitude, N_electrodes):

        for frequency in [1e5,2e5,3e5,4e5,5e5]:

            for inner_stat in range(5):

                N                           = 3
                voltages_values             = amplitude*np.cos(2*np.pi*frequency*time_steps)
                topology_parameter          = {}
                topology_parameter["Nx"]    = N
                topology_parameter["Ny"]    = N
                topology_parameter["Nz"]    = 1
                # topology_parameter["e_pos"] = [[0,0,0], [int((N-1)/2),0,0], [N-1,0,0], [0,int((N-1)/2),0], [0,N-1,0], [N-1,int((N)/2),0], [int((N)/2),(N-1),0], [N-1,N-1,0]]
                topology_parameter["e_pos"] = [[0,0,0], [N-1,N-1,0]]

                target_electrode    = len(topology_parameter["e_pos"]) - 1
                # folder              = f"/home/jonas/phd/NanoNets/test_runs/time_run/data/periodic/dis_f_01_"#"/scratch/tmp/j_mens07/data/system_size/"
                folder              = f"/mnt/c/Users/jonas/Desktop/phd/NanoNets/test_runs/time_run/data/periodic/f={np.round(frequency)}_A={amplitude}"#"/scratch/tmp/j_mens07/data/system_size/"
                voltages            = pd.DataFrame(np.zeros((N_voltages, N_electrodes+1)))
                voltages.iloc[:,0]  = voltages_values
                
                np_info = {
                    "eps_r"         : 2.6,
                    "eps_s"         : 3.9,
                    "mean_radius"   : 10.0,
                    "std_radius"    : 0.0,
                    "np_distance"   : 1.0
                }

                sim_class   = nanonets.simulation(voltages.values)
                sim_class.init_cubic(folder, topology_parameter, add_to_path=f"_{thread}_{inner_stat}", np_info=np_info)
                sim_class.run_var_voltages(target_electrode, time_steps, save_th=1)
                # model.time_simulation(target_electrode, time_steps, topology_parameter, voltages.values, folder, add_to_path=f"_{thread}_f{f_mult}",
                #                             save_th=1, T_val=0.0)

    
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, N_voltages, time_steps, amplitude, N_electrodes))
        process.start()