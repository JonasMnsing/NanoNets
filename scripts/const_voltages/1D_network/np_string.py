# Libraries
import numpy as np
import pandas as pd
# import multiprocessing

# Extend PATH Variable
import sys
sys.path.append("src/")

# NanoNets Simulation Tool
import nanonets
import nanonets_utils

def parallel_code(thread, voltages):

    N_particles             = 5
    network_topology        = "cubic"

    # Topology
    # if thread == 0:
    #     topology_parameter    = {
    #         "Nx"    : N_particles,
    #         "Ny"    : 1,
    #         "Nz"    : 1,
    #         "e_pos" : [[0,0,0],[2,0,0],[4,0,0]]
    #     }
    # else:
    #     topology_parameter    = {
    #         "Nx"    : N_particles,
    #         "Ny"    : 1,
    #         "Nz"    : 1,
    #         "e_pos" : [[0,0,0],[3,0,0],[4,0,0]]
    #     }
    
    thread = 1
    topology_parameter    = {
        "Nx"    : N_particles,
        "Ny"    : 1,
        "Nz"    : 1,
        "e_pos" : [[0,0,0],[3,0,0],[4,0,0]]
    }

    # Run Simulation
    np_network_sim = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, folder=f"scripts/const_voltages/1D_network/data/", add_to_path=f'_{thread}')
    np_network_sim.run_const_voltages(voltages=voltages, target_electrode=2, save_th=10)

if __name__ == '__main__':

    # Electrode Voltages
    N_voltages  = 2000
    volt_array  = np.array([0,0,0,0])

    for V in [-0.06, -0.05, -0.02, 0.02, 0.05, 0.06]:

        voltages        = np.zeros(shape=(N_voltages,4))
        voltages[:,0]   = np.linspace(-0.2,0.2,N_voltages)
        voltages[:,1]   = V
        volt_array      = np.vstack((volt_array, voltages))
    
    parallel_code(0, volt_array[1:,:])

    # for i in range(2):

    #     process = multiprocessing.Process(target=parallel_code, args=(0,volt_array[1:,:]))
    #     process.start()



