import numpy as np
import sys
sys.path.append("../../../src/")

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

        nanonets_utils.memory_capacity_simulation(time_steps, voltages, train_length=5000, test_length=5000, remember_distance=rem_d,
                                                  folder=folder, np_info=np_info, network_topology=network_topology,
                                                  topology_parameter=topology_parameter, save_th=0.1, path_info=f'_{thread}', R=R, Rstd=Rstd)
        
if __name__ == '__main__':

    # Define Time Scale
    step_size   = 1.5e-9
    max_time    = 1e-4
    time_steps  = np.arange(0,max_time,step_size)
    
    # Network Style
    network_topology    = "cubic"
    N_d                 = 7 
    topology_parameter  = {
        "Nx"    : N_d,
        "Ny"    : N_d,
        "Nz"    : 1,
        "e_pos" : [[0,0,0],[3,0,0],[0,3,0],[6,0,0],[0,6,0],[6,3,0],[3,6,0],[6,6,0]]
    }

    # Simulation
    N_processes = 10
    folder      = "scripts/ipynb/memory_capacity/data/lattice_R_disorder/r1/"

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,time_steps,network_topology,topology_parameter,None,folder,25,5))
        process.start()
    
    folder      = "scripts/ipynb/memory_capacity/data/lattice_R_disorder/r2/"

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,time_steps,network_topology,topology_parameter,None,folder,25,10))
        process.start()
    
    folder      = "scripts/ipynb/memory_capacity/data/lattice_R_disorder/r3/"

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,time_steps,network_topology,topology_parameter,None,folder,25,15))
        process.start()