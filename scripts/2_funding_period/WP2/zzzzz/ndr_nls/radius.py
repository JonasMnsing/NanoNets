"""
Run KMC Code for const set of voltages with fixed parameters defined in "params.csv" @ folder.
Output Electrode @ last position in topology_parameter key "pos"
"""

import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets
import nanonets_utils
import multiprocessing

# Simulation Function
def parallel_code(thread, rows, voltages, topology_parameter, folder, np_info, res_info):

    target_electrode    = len(topology_parameter["e_pos"]) - 1
    thread_rows         = rows[thread]
        
    sim_class = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter, folder=folder+"data/",
                                    res_info=res_info, np_info=np_info, seed=thread, add_to_path=f"_th={thread}")
    sim_class.run_const_voltages(voltages=voltages[thread_rows,:], target_electrode=target_electrode)

if __name__ == '__main__':

    # Number of voltages and CPU processes
    N_voltages  = 80640
    N_processes = 36

    # Maximum absolute voltage values
    Vc  = 0.05
    Vg  = 0#0.2
    Vi  = 0.01

    # Generate voltage values      
    v_rand      = np.repeat(np.random.uniform(low=-Vc, high=Vc, size=((int(N_voltages/4),5))), 4, axis=0)
    v_gates     = np.repeat(np.random.uniform(low=-Vg, high=Vg, size=int(N_voltages/4)),4)
    i1          = np.tile([0.0,0.0,Vi,Vi], int(N_voltages/4))
    i2          = np.tile([0.0,Vi,0.0,Vi], int(N_voltages/4))

    # Voltages
    voltages        = np.zeros(shape=(N_voltages,9))
    voltages[:,0]   = v_rand[:,0]
    voltages[:,1]   = i1
    voltages[:,2]   = v_rand[:,1]
    voltages[:,3]   = i2
    voltages[:,4]   = v_rand[:,2]
    voltages[:,5]   = v_rand[:,3]
    voltages[:,6]   = v_rand[:,4]
    voltages[:,-1]  = v_gates

    topology_parameter  = {
        "Nx"    :   7,
        "Ny"    :   7,
        "Nz"    :   1,
        "e_pos" :   [[0,0,0],[3,0,0],[6,0,0],[0,3,0],[0,6,0],[6,3,0],[3,6,0],[6,6,0]],
    }
    np_info = {
        "eps_r"         : 2.6,
        "eps_s"         : 3.9,
        "mean_radius"   : 10.0,
        "std_radius"    : 2.0,
        "np_distance"   : 1.0
    }
    res_info = {
        "mean_R"    : 25.0,
        "std_R"     : 0.0,
        "dynamic"   : False
    }

    folder  = "/scratch/tmp/j_mens07/data/ndr_nls_disorder/radius/"
    rows    = [np.arange(j*N_voltages/N_processes,(j+1)*N_voltages/N_processes, dtype=int) for j in range(N_processes)]

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, rows, voltages, topology_parameter, folder,
                                                                      np_info, res_info))
        process.start()