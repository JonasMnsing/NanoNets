"""
Phase space sample of a 9 x 9 lattice network.
"""

import numpy as np
import sys

# Add to path
sys.path.append("/home/j/j_mens07/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets
import nanonets_utils
import multiprocessing

# Simulation Function
def parallel_code(thread, rows, voltages):

    target_electrode    = len(topology_parameter["e_pos"]) - 1
    thread_rows         = rows[thread]
    
    sim_class = nanonets.simulation(topology_parameter=topology_parameter, folder=folder)
    sim_class.run_const_voltages(voltages=voltages[thread_rows,:], target_electrode=target_electrode, save_th=20)

if __name__ == '__main__':

    # N_p x N_p Values for Network Size 
    N_p = 9

    # Topology Parameter
    topology_parameter          = {
        "Nx"                :   N_p,
        "Ny"                :   N_p,
        "Nz"                :   1,
        "e_pos"             :   [[0,0,0], [int((N_p-1)/2),0,0], [N_p-1,0,0], 
                                [0,int((N_p-1)/2),0], [0,N_p-1,0], [N_p-1,int((N_p)/2),0],
                                [int((N_p)/2),(N_p-1),0], [N_p-1,N_p-1,0]],
        "electrode_type"    :   ['constant','constant','constant','constant','constant','constant','constant','floating']
    }

    if topology_parameter["electrode_type"][-1] == "constant":
        folder  = "/home/j/j_mens07/phd/data/1_funding_period/current/surrogate_model/"
    else:
        folder  = "/home/j/j_mens07/phd/data/1_funding_period/potential/surrogate_model/"

    # Number of voltages and CPU processes
    N_voltages  = 100000 #80640
    N_processes = 10 #36

    # Voltage values
    U_e         = 0.1
    voltages    = nanonets_utils.lhs_sample(U_e=U_e, N_samples=N_voltages, topology_parameter=topology_parameter)
    
    # Simulated rows for each process
    index   = [i for i in range(N_voltages)]
    rows    = [index[i::N_processes] for i in range(N_processes)]

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, rows, voltages))
        process.start()