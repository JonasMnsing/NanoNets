"""
Influence of variable system size on Boolean Logic Functionality. Simulation covers N x N lattice networks.
The sampled phase space, i.e. Control and Input Voltage range remains const.
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
def parallel_code(thread, rows, U_e, N_voltages, N_p_min, N_p_max):

    for N in range(N_p_min, N_p_max+1):
        
        # Topology Parameter
        topology_parameter          = {
            "Nx"                :   N,
            "Ny"                :   N,
            "Nz"                :   1,
            "e_pos"             :   [[0,0,0], [int((N-1)/2),0,0], [N-1,0,0], 
                                    [0,int((N-1)/2),0], [0,N-1,0], [N-1,int((N)/2),0],
                                    [int((N)/2),(N-1),0], [N-1,N-1,0]],
            "electrode_type"    :   ['constant','constant','constant','constant','constant','constant','constant','constant']
        }

        if topology_parameter["electrode_type"][-1] == "constant":
            folder  = "/home/j/j_mens07/phd/data/1_funding_period/current/system_size/"
        else:
            folder  = "/home/j/j_mens07/phd/data/1_funding_period/potential/system_size/"

        voltages            = nanonets_utils.logic_gate_sample(U_e=U_e, input_pos=[1,3], N_samples=N_voltages, topology_parameter=topology_parameter)
        target_electrode    = len(topology_parameter["e_pos"]) - 1
        thread_rows         = rows[thread]
        
        sim_class = nanonets.simulation(topology_parameter=topology_parameter, folder=folder)
        sim_class.run_const_voltages(voltages=voltages[thread_rows,:], target_electrode=target_electrode, save_th=20)

if __name__ == '__main__':

    # N_p x N_p Values for Network Size 
    N_p_min = 9
    N_p_max = 16

    # Number of voltages and CPU processes
    N_voltages  = 20000 #80640
    N_processes = 10 #36

    # Maximum absolute voltage values
    U_e = 0.1
    
    # Simulated rows for each process
    index   = [i for i in range(N_voltages)]
    rows    = [index[i::N_processes] for i in range(N_processes)]

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, rows, U_e, N_voltages, N_p_min, N_p_max))
        process.start()