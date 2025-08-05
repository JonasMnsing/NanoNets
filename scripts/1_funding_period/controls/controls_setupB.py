"""
Influence of variable numbers of control electrodes on Boolean Logic Functionality. The system size is fixed at 9x9. 
Simulation adds controls from the top right to bottom left corner.
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
def parallel_code(thread, rows, U_e, N_voltages, N_c):

    for N in N_c:
        
        # Topology Parameter
        topology_parameter          = {}
        topology_parameter["Nx"]    = 9
        topology_parameter["Ny"]    = 9
        topology_parameter["Nz"]    = 1

        positions       = [[0,0,0], [4,0,0], [0,4,0]]

        if N >= 3:
            positions.append([8,6,0])
            positions.append([6,8,0])
        if N >= 5:
            positions.append([8,4,0]) 
            positions.append([4,8,0])                
        if N >= 7:
            positions.append([8,0,0]) 
            positions.append([0,8,0])
        if N >= 9:
            positions.append([6,0,0])
            positions.append([0,6,0])  

        positions.append([8,8,0])

        topology_parameter["e_pos"]                 = positions
        topology_parameter["electrode_type"]        = ['constant' for _ in range(len(positions))]
        topology_parameter["electrode_type"][-1]    = 'floating'

        target_electrode    = len(topology_parameter["e_pos"]) - 1
        thread_rows         = rows[thread]
        voltages            = nanonets_utils.logic_gate_sample(U_e=U_e, input_pos=[1,3], N_samples=N_voltages, topology_parameter=topology_parameter)

        if topology_parameter["electrode_type"][-1] == "constant":
            folder  = "/home/j/j_mens07/phd/data/1_funding_period/current/electrode_pos/setupB/"
        else:
            folder  = "/home/j/j_mens07/phd/data/1_funding_period/potential/electrode_pos/setupB/"

        voltages            = nanonets_utils.logic_gate_sample(U_e=U_e, input_pos=[1,2], N_samples=N_voltages, topology_parameter=topology_parameter)
        target_electrode    = len(topology_parameter["e_pos"]) - 1
        thread_rows         = rows[thread]
        
        sim_class = nanonets.simulation(topology_parameter=topology_parameter, folder=folder)
        sim_class.run_const_voltages(voltages=voltages[thread_rows,:], target_electrode=target_electrode, save_th=20)

if __name__ == '__main__':

    # Number of Controls 
    N_c = [1,3,5,7,9]

    # Number of voltages and CPU processes
    N_voltages  = 20000 #80640
    N_processes = 10 #36

    # Maximum absolute voltage values
    U_e = 0.1
    
    # Simulated rows for each process
    index   = [i for i in range(N_voltages)]
    rows    = [index[i::N_processes] for i in range(N_processes)]

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, rows, U_e, N_voltages, N_c))
        process.start()