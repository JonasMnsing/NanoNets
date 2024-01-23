import numpy as np
import pandas as pd
import sys
sys.path.append("/home/j/j_mens07/NanoNets/src/")
import nanonets
import multiprocessing

if __name__ == '__main__':

    ##############################################
    # INIT PARAMETER AND CORRESPONDING DICTONARIES
    ##############################################

    N_voltages      = 80028
    N_processes     = 36
    N_values        = [13,11,9,7,5,3,1]
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
            topology_parameter["Nx"]    = 5
            topology_parameter["Ny"]    = 5
            topology_parameter["Nz"]    = 1
            tunnel_order                = 1

            voltages        = pd.DataFrame(np.zeros((N_voltages,N+4)))  
            positions       = [[topology_parameter["Nx"]-1,0,0],[0,topology_parameter["Ny"]-1,0]]

            voltages.iloc[:,0]   = i1 
            voltages.iloc[:,1]   = i2
            voltages.iloc[:,-1]  = v_gates 

            if N >= 1:
                positions.append([0,0,0])
                voltages.iloc[:,2] = v_rand[:,0]
            if N >= 3:
                positions.append([topology_parameter["Nx"]-4,0,0])
                positions.append([0,topology_parameter["Ny"]-4,0])  
                voltages.iloc[:,3] = v_rand[:,1]
                voltages.iloc[:,4] = v_rand[:,2]
            if N >= 5:
                positions.append([topology_parameter["Nx"]-3,0,0])
                positions.append([0,topology_parameter["Ny"]-3,0])    
                voltages.iloc[:,5] = v_rand[:,3]
                voltages.iloc[:,6] = v_rand[:,4]                             
            if N >= 7:
                positions.append([topology_parameter["Nx"]-2,0,0])
                positions.append([0,topology_parameter["Ny"]-2,0])  
                voltages.iloc[:,7] = v_rand[:,5]
                voltages.iloc[:,8] = v_rand[:,6]                                
            if N >= 9:
                positions.append([topology_parameter["Nx"]-1,topology_parameter["Ny"]-4,0])
                positions.append([topology_parameter["Nx"]-4,topology_parameter["Ny"]-1,0]) 
                voltages.iloc[:,9] = v_rand[:,7]
                voltages.iloc[:,10] = v_rand[:,8]                 
            if N >= 11:
                positions.append([topology_parameter["Nx"]-1,topology_parameter["Ny"]-3,0])
                positions.append([topology_parameter["Nx"]-3,topology_parameter["Ny"]-1,0]) 
                voltages.iloc[:,11] = v_rand[:,9]
                voltages.iloc[:,12] = v_rand[:,10]               
            if N >= 13:
                positions.append([topology_parameter["Nx"]-1,topology_parameter["Ny"]-2,0])
                positions.append([topology_parameter["Nx"]-2,topology_parameter["Ny"]-1,0])
                voltages.iloc[:,13] = v_rand[:,11]
                voltages.iloc[:,14] = v_rand[:,12]                            
            
            positions.append([topology_parameter["Nx"]-1, topology_parameter["Ny"]-1, 0])

            topology_parameter["e_pos"] = positions
            target_electrode            = len(topology_parameter["e_pos"]) - 1
            folder                      = "/scratch/tmp/j_mens07/data/controls_c_to_o/"

            sim_dic                 = {}
            sim_dic['error_th']     = 0.05
            sim_dic['max_jumps']    = 10000000

            # folder          = "../data/controls_c_to_o/"
            folder          = "/scratch/tmp/j_mens07/data/controls_c_to_o/"
            simulation_info = {
                "p_eq"              : 0.95,
                "max_pot_dev"       : 0.1,
                "max_jumps"         : 5000000,
                "max_rel_error"     : 0.05,
                "thread_num"        : thread
            }
            nanonets.cubic_net_simulation(target_electrode, topology_parameter, voltages.values[thread_rows,:], folder,
                                save_th=100, tunnel_order=tunnel_order, T_val=0, sim_dic=sim_dic)
    
    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,rows,v_rand,v_gates, i1, i2, N_voltages))
        process.start()
