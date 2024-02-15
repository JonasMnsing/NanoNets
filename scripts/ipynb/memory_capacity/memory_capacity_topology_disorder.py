import numpy as np
import sys
sys.path.append("src/")

# NanoNets Simulation Tool
import nanonets
import nanonets_utils

import multiprocessing

def parallel_code(thread, time_steps, network_topology, topology_parameter,  np_info, folder, R, Rstd):

    rng         = np.random.default_rng()
    N_steps     = len(time_steps)
    input_volt  = rng.normal(0.0,1.0,N_steps)

    # Voltage Array
    voltages        = np.zeros((N_steps, 9))
    voltages        = np.zeros((N_steps, 9))
    voltages[:,0]   = input_volt

    for rem_d in np.arange(0,26,1):

        nanonets_utils.memory_capacity_simulation(time_steps, voltages, train_length=5000, test_length=5000,
                                                remember_distance=rem_d, folder=folder, np_info=np_info,
                                                network_topology=network_topology, topology_parameter=topology_parameter,
                                                save_th=0.1, path_info=f'_{thread}', R=R, Rstd=Rstd)
        
if __name__ == '__main__':

    # Define Time Scale
    step_size   = 1.5e-9
    max_time    = 1e-4
    time_steps  = np.arange(0,max_time,step_size)
    
    # Network Style
    network_topology    = "random"
    topology_parameter  = {
        "Np"    : 49,
        "Nj"    : 4,
        "e_pos" : [[-1,-1],[0,-1],[1,-1],[-1,0],[-1,1],[0,1],[1,0],[1,1]]
    }

    # Simulation Parameter
    N_processes = 10
    folder      = "scripts/ipynb/memory_capacity/data/disordered_topology/"

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,time_steps,network_topology,topology_parameter,None,folder,25,0))
        process.start()