import numpy as np
import pandas as pd
import sys
# sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
sys.path.append("/home/jonas/phd/NanoNets/src/")
import model
import multiprocessing

if __name__ == '__main__':

    ##############################################
    # INIT PARAMETER AND CORRESPONDING DICTONARIES
    ##############################################

    N_voltages      = 80028
    N_processes     = 36
    N_values        = [2,4,6,8,9]
    v_rand          = np.repeat(np.random.uniform(low=-0.05, high=0.05, size=((int(N_voltages/4),max(N_values)))),4,axis=0)
    v_gates         = np.repeat(np.random.uniform(low=-0.1, high=0.1, size=int(N_voltages/4)),4)
    i1              = np.tile([0.0,0.0,0.01,0.01],int(N_voltages/4))
    i2              = np.tile([0.0,0.01,0.0,0.01],int(N_voltages/4))
    index           = [i for i in range(N_voltages)]
    rows            = [index[i::N_processes] for i in range(N_processes)]

    def parallel_code(thread, rows, v_rand, v_gates, i1, i2, N_voltages, N_values):

        thread_rows = rows[thread]

        for N in N_values:

            topology_parameter          = {}
            topology_parameter["Nx"]    = 7
            topology_parameter["Ny"]    = 7
            topology_parameter["Nz"]    = 1

            voltages            = pd.DataFrame(np.zeros((N_voltages,N+4)))  
            positions           = [[topology_parameter["Nx"]-1,topology_parameter["Ny"]-3,0], [topology_parameter["Nx"]-3,topology_parameter["Ny"]-1,0]]
            voltages.iloc[:,0]  = i1
            voltages.iloc[:,1]  = i2
            voltages.iloc[:,-1] = v_gates 

            if N >= 2:
                positions.append([topology_parameter["Nx"]-1,topology_parameter["Ny"]-5,0])
                positions.append([topology_parameter["Nx"]-5,topology_parameter["Ny"]-1,0])
                voltages.iloc[:,2] = v_rand[:,0]
                voltages.iloc[:,3] = v_rand[:,1]
            if N >= 4:
                positions.append([topology_parameter["Nx"]-1,0,0]) 
                positions.append([0,topology_parameter["Ny"]-1,0])
                voltages.iloc[:,4] = v_rand[:,2]
                voltages.iloc[:,5] = v_rand[:,3]                
            if N >= 6:
                positions.append([topology_parameter["Nx"]-3,0,0]) 
                positions.append([0,topology_parameter["Ny"]-3,0])
                voltages.iloc[:,6] = v_rand[:,4]
                voltages.iloc[:,7] = v_rand[:,5]                           
            if N >= 8:
                positions.append([topology_parameter["Nx"]-5,0,0])
                positions.append([0,topology_parameter["Ny"]-5,0])  
                voltages.iloc[:,8] = v_rand[:,6]
                voltages.iloc[:,9] = v_rand[:,7]
            if N >= 9:
                positions.append([0,0,0])
                voltages.iloc[:,10] = v_rand[:,8]

            positions.append([topology_parameter["Nx"]-1, topology_parameter["Ny"]-1, 0])

            topology_parameter["e_pos"] = positions
            target_electrode            = len(topology_parameter["e_pos"]) - 1
            folder                      = "/scratch/tmp/j_mens07/data/c1/"

            sim_dic                 = {}
            sim_dic['error_th']     = 0.05
            sim_dic['max_jumps']    = 10000000

            sim_class = model.simulation(voltages.values[thread_rows,:])
            sim_class.init_cubic(folder=folder, topology_parameter=topology_parameter)
            sim_class.run_const_voltages(target_electrode=target_electrode, sim_dic=sim_dic, save_th=1)
    
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,rows,v_rand,v_gates, i1, i2, N_voltages, N_values))
        process.start()
