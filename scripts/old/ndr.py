import numpy as np
import pandas as pd
import sys
sys.path.append("/home/jonas/phd/NanoNets/src/")
# sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src")
import simulation
import voltage_config
import multiprocessing

if __name__ == '__main__':

    # Voltage Sweep
    N_voltages          = 10000
    N_electrodes        = 9
    N_processes         = 10

    v_rand              = voltage_config.ndr_config(low=-0.05, high=0.05, sweep_low=-0.01,
                                                    sweep_high=0.01, sweep_col=0, N_sweep=50,
                                                    N_rows=N_voltages, N_cols=(N_electrodes-1))
    v_gates             = voltage_config.ndr_config_G(low=-0.1, high=0.1, N_sweep=50, N_rows=N_voltages)
    index               = [i for i in range(len(v_gates))]
    rows                = [index[i::N_processes] for i in range(N_processes)]

    print(v_rand)

    def parallel_code(thread, rows, v_rand, v_gates, N_voltages, N_electrodes):

        for N in [5]:
            
            topology_parameter          = {}
            topology_parameter["Nx"]    = N
            topology_parameter["Ny"]    = N
            topology_parameter["Nz"]    = 1
            topology_parameter["e_pos"] = [[0,0,0], [int((N-1)/2),0,0], [N-1,0,0], [0,int((N-1)/2),0], [0,N-1,0], [N-1,int((N-1)/2),0], [int((N-1)/2),(N-1),0], [N-1,N-1,0]]
            tunnel_order                = 1

            sim_dic                 = {}
            sim_dic['error_th']     = 0.05
            sim_dic['max_jumps']    = 5000000

            target_electrode    = len(topology_parameter["e_pos"]) - 1
            folder              = "data/"
            thread_rows         = rows[thread]
            voltages            = pd.DataFrame(np.zeros((N_voltages,N_electrodes)))

            for i in range(N_electrodes-1):
                voltages.iloc[:,i]  = v_rand[:,i]
            voltages.iloc[:,-1] = v_gates
            
            simulation.cubic_net_simulation(target_electrode, topology_parameter, voltages.values[thread_rows,:], folder,
                                save_th=10, tunnel_order=tunnel_order, T_val=0, sim_dic=sim_dic)
    

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,rows, v_rand, v_gates, N_voltages, N_electrodes))
        process.start()