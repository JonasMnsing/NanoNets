import numpy as np
import pandas as pd
import sys
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
import nanonets
import multiprocessing

if __name__ == '__main__':

    # Voltage Sweep
    N_voltages          = 80640
    N_processes         = 36
    v_rand              = np.repeat(np.random.uniform(low=-0.05, high=0.05, size=((int(N_voltages/4),5))), 4, axis=0)
    v_gates             = np.repeat(np.random.uniform(low=-0.1, high=0.1, size=int(N_voltages/4)),4)
    i1                  = np.tile([0.0,0.0,0.01,0.01], int(N_voltages/4))
    i2                  = np.tile([0.0,0.01,0.0,0.01], int(N_voltages/4))
    index               = [i for i in range(N_voltages)]
    rows                = [index[i::N_processes] for i in range(N_processes)]

    def parallel_code(thread, rows, v_rand, v_gates, i1, i2, N_voltages):

        for N in range(3,11):
            
            topology_parameter          = {}
            topology_parameter["Nx"]    = N
            topology_parameter["Ny"]    = N
            topology_parameter["Nz"]    = 1
            topology_parameter["e_pos"] = [[0,0,0], [int((N-1)/2),0,0], [N-1,0,0], [0,int((N-1)/2),0], [0,N-1,0], [N-1,int((N)/2),0], [int((N)/2),(N-1),0], [N-1,N-1,0]]

            sim_dic                 = {}
            sim_dic['error_th']     = 0.05
            sim_dic['max_jumps']    = 10000000

            target_electrode    = len(topology_parameter["e_pos"]) - 1
            folder              = "/scratch/tmp/j_mens07/data/system_size_new/"
            thread_rows         = rows[thread]
            voltages            = pd.DataFrame(np.zeros((N_voltages,9)))
            voltages.iloc[:,0]  = v_rand[:,0]
            voltages.iloc[:,1]  = i1
            voltages.iloc[:,2]  = v_rand[:,1]
            voltages.iloc[:,3]  = i2
            voltages.iloc[:,4]  = v_rand[:,2]
            voltages.iloc[:,5]  = v_rand[:,3]
            voltages.iloc[:,6]  = v_rand[:,4]
            voltages.iloc[:,-1] = v_gates

            sim_class = nanonets.simulation(voltages.values[thread_rows,:])
            sim_class.init_cubic(folder=folder, topology_parameter=topology_parameter)
            sim_class.run_const_voltages(target_electrode=target_electrode, sim_dic=sim_dic, save_th=20, T_val=0.0)

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,rows,v_rand,v_gates, i1, i2, N_voltages))
        process.start()