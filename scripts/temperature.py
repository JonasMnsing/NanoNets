import numpy as np
import pandas as pd
import sys
sys.path.append("/home/j/j_mens07/NanoNets/src/")
sys.path.append("/home/jonas/phd/NanoNets/src/")
import model
import voltage_config
import multiprocessing

if __name__ == '__main__':

    # Voltage Sweep
    N_voltages      = 40000
    N_electrodes    = 9
    N_processes     = 5
    v_rand          = voltage_config.logic_gate_config(low=-0.05, high=0.05, off_state=0.0, on_state=0.01, i1_col=5, i2_col=6, N_rows=N_voltages, N_cols=(N_electrodes-1))
    v_gates         = voltage_config.logic_gate_config_G(low=-0.1, high=0.1, N_rows=N_voltages)
    index           = [i for i in range(N_voltages)]
    rows            = [index[i::N_processes] for i in range(N_processes)]

    def parallel_code(thread, rows, v_rand, v_gates, N_voltages, N_electrodes):

        for Temp in [2,4,8,16,32,64]:
            
            topology_parameter          = {}
            topology_parameter["Nx"]    = 5
            topology_parameter["Ny"]    = 5
            topology_parameter["Nz"]    = 1
            topology_parameter["e_pos"] = [[0,0,0], [int((5-1)/2),0,0], [5-1,0,0], [0,int((5-1)/2),0], [0,5-1,0], [5-1,int((5-1)/2),0], [int((5-1)/2),(5-1),0], [5-1,5-1,0]]
            tunnel_order                = 1

            sim_dic                 = {}
            sim_dic['error_th']     = 0.05
            sim_dic['max_jumps']    = 10000000

            target_electrode    = len(topology_parameter["e_pos"]) - 1
            folder              = f"/home/jonas/phd/data/temperature/T={Temp}_"
            thread_rows         = rows[thread]
            voltages            = pd.DataFrame(np.zeros((N_voltages, N_electrodes)))

            for i in range(N_electrodes-1):
                voltages.iloc[:,i]  = v_rand[:,i]
            voltages.iloc[:,-1] = v_gates

            model.cubic_net_simulation(target_electrode, topology_parameter, voltages.values[thread_rows,:], folder,
                                save_th=10, tunnel_order=tunnel_order, T_val=Temp, sim_dic=sim_dic)
    
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,rows,v_rand,v_gates, N_voltages, N_electrodes))
        process.start()