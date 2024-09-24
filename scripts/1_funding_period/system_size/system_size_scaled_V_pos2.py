"""
Influence of variable system size on Boolean Logic Functionality. Simulation covers N x N lattice networks.
The sampled phase space, i.e. Control and Input Voltage range is scales with system size (see scales var).
"""

import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets
import multiprocessing

# Simulation Function
def parallel_code(thread, rows, v_rand, v_gates, i1, i2, N_voltages, scales, N_p_min, N_p_max, sim_dic, folder):

    for i, N in enumerate(range(N_p_min, N_p_max)):
        
        # Topology Parameter
        topology_parameter          = {}
        topology_parameter["Nx"]    = N
        topology_parameter["Ny"]    = N
        topology_parameter["Nz"]    = 1
        topology_parameter["e_pos"] = [[0,0,0], [1,0,0], [N-1,0,0], 
                                        [0,1,0], [0,N-1,0], [N-1,N-2,0],
                                        [N-2,(N-1),0], [N-1,N-1,0]]
                
        # Voltage Values
        voltages        = np.zeros(shape=(N_voltages,9))
        voltages[:,0]   = scales[i] * v_rand[:,0]
        voltages[:,1]   = scales[i] * i1
        voltages[:,2]   = scales[i] * v_rand[:,1]
        voltages[:,3]   = scales[i] * i2
        voltages[:,4]   = scales[i] * v_rand[:,2]
        voltages[:,5]   = scales[i] * v_rand[:,3]
        voltages[:,6]   = scales[i] * v_rand[:,4]
        voltages[:,-1]  = v_gates

        target_electrode    = len(topology_parameter["e_pos"]) - 1
        thread_rows         = rows[thread]
        
        sim_class = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter, folder=folder)
        sim_class.run_const_voltages(voltages=voltages[thread_rows,:], target_electrode=target_electrode, sim_dic=sim_dic, save_th=20)

if __name__ == '__main__':

    # N_p x N_p Values for Network Size 
    N_p_min = 3
    N_p_max = 18

    # Number of voltages and CPU processes
    N_voltages  = 80000 #80640
    N_processes = 10 #36

    # Maximum absolute voltage values
    Vc  = 0.05
    Vg  = 0.1
    Vi  = 0.01

    # Generate voltage values      
    v_rand      = np.repeat(np.random.uniform(low=-Vc, high=Vc, size=((int(N_voltages/4),5))), 4, axis=0)
    v_gates     = np.repeat(np.random.uniform(low=-Vg, high=Vg, size=int(N_voltages/4)),4)
    i1          = np.tile([0.0,0.0,Vi,Vi], int(N_voltages/4))
    i2          = np.tile([0.0,Vi,0.0,Vi], int(N_voltages/4))
    scales      = [0.69, 0.77, 0.84, 0.92, 1., 1.08, 1.16, 1.23, 1.31, 1.39, 1.47, 1.55, 1.62, 1.7, 1.78, 1.86]

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
    # folder  = "/scratch/tmp/j_mens07/data/system_size/"
    folder  = "/home/jonas/phd/data/system_size/scaled_pos2/"

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, rows, v_rand, v_gates, i1,
                                                                      i2, N_voltages, scales, N_p_min,
                                                                      N_p_max, sim_dic, folder))
        process.start()