# Libraries
import numpy as np
import pandas as pd
import multiprocessing

# Extend PATH Variable
import sys
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

# NanoNets Simulation Tool
import nanonets
import nanonets_utils

def parallel_code(thread, voltages):

    N_particles             = 5
    network_topology        = "cubic"

    # Topology
    if thread == 0:
        topology_parameter    = {
            "Nx"    : N_particles,
            "Ny"    : 1,
            "Nz"    : 1,
            "e_pos" : [[0,0,0],[2,0,0],[4,0,0]]
        }
    else:
        topology_parameter    = {
            "Nx"    : N_particles,
            "Ny"    : 1,
            "Nz"    : 1,
            "e_pos" : [[0,0,0],[3,0,0],[4,0,0]]
        }

    # Run Simulation
    np_network_sim = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, folder=f"scripts/const_voltages/1D_network/data/c_", add_to_path=f'_{thread}')
    np_network_sim.run_const_voltages(voltages=voltages, target_electrode=2, save_th=10)

if __name__ == '__main__':

    # Electrode Voltages
    N_voltages      = 1000
    voltages        = np.zeros(shape=(N_voltages,4))
    voltages[:,1]   = np.linspace(-0.1,0.1,N_voltages)
    voltages        = np.vstack((voltages,voltages))
    voltages        = np.vstack((voltages,voltages))

    voltages[N_voltages:2*N_voltages,0] = 0.01
    voltages[2*N_voltages:,0]           = 0.03

    for i in range(2):

        process = multiprocessing.Process(target=parallel_code, args=(i,voltages))
        process.start()



