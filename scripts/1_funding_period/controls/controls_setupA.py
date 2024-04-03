"""
Influence of variable numbers of control electrodes on Boolean Logic Functionality. The system size is fixed at 7x7. 
Simulation adds controls from the bottom left to top right corner.
"""

import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets
import multiprocessing

# Simulation Function
def parallel_code(thread, rows, v_rand, v_gates, i1, i2, N_voltages, N_c, sim_dic, folder):

    for N in N_c:
        
        # Topology Parameter
        topology_parameter          = {}
        topology_parameter["Nx"]    = 7
        topology_parameter["Ny"]    = 7
        topology_parameter["Nz"]    = 1

        positions       = [[0,0,0], [2,0,0], [0,2,0]]
        voltages        = np.zeros(shape=(N_voltages,N+4))
        voltages[:,0]   = v_rand[:,0]
        voltages[:,1]   = i1
        voltages[:,2]   = i2
        voltages[:,-1]  = v_gates

        if N >= 3:
            positions.append([4,0,0])
            positions.append([0,4,0])
            voltages[:,3] = v_rand[:,1]
            voltages[:,4] = v_rand[:,2]
        if N >= 5:
            positions.append([6,0,0]) 
            positions.append([0,6,0])
            voltages[:,5] = v_rand[:,3]
            voltages[:,6] = v_rand[:,4]                
        if N >= 7:
            positions.append([6,2,0]) 
            positions.append([2,6,0])
            voltages[:,7] = v_rand[:,5]
            voltages[:,8] = v_rand[:,6]                           
        if N >= 9:
            positions.append([6,4,0])
            positions.append([4,6,0])  
            voltages[:,9] = v_rand[:,7]
            voltages[:,10] = v_rand[:,8]
        
        positions.append([6,6,0])

        topology_parameter["e_pos"] = positions
        target_electrode            = len(topology_parameter["e_pos"]) - 1
        thread_rows                 = rows[thread]
        
        sim_class = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter, folder=folder)
        sim_class.run_const_voltages(voltages=voltages[thread_rows,:], target_electrode=target_electrode, sim_dic=sim_dic, save_th=20)

if __name__ == '__main__':

    # Number of Controls 
    N_c = [3,5,7,9]

    # Number of voltages and CPU processes
    N_voltages  = 80000 #80640
    N_processes = 10 #36

    # Maximum absolute voltage values
    Vc  = 0.05
    Vg  = 0.1
    Vi  = 0.01

    # Generate voltage values      
    v_rand      = np.repeat(np.random.uniform(low=-Vc, high=Vc, size=((int(N_voltages/4),max(N_c)))), 4, axis=0)
    v_gates     = np.repeat(np.random.uniform(low=-Vg, high=Vg, size=int(N_voltages/4)),4)
    i1          = np.tile([0.0,0.0,Vi,Vi], int(N_voltages/4))
    i2          = np.tile([0.0,Vi,0.0,Vi], int(N_voltages/4))
    
    # Simulated rows for each process
    index   = [i for i in range(N_voltages)]
    rows    = [index[i::N_processes] for i in range(N_processes)]

    # Simulation Parameter
    sim_dic = {
        'error_th'  :   0.05,
        'max_jumps' :   10000000,
        'eq_steps'  :   100000
    }

    # Save Folder
    # folder  = "/scratch/tmp/j_mens07/data/controls_setupA/"
    folder  = "/home/jonas/phd/data/controls_setupA/"

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, rows, v_rand, v_gates, i1, i2,
                                                                      N_voltages, N_c, sim_dic, folder))
        process.start()