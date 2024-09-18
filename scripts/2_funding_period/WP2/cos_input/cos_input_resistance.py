import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets
import multiprocessing

# Simulation Function
def run_simulation(freq, time_steps, network_topology, topology_parameter, eq_steps, folder, stat_size, seed, amplitude, N_voltages, R_val, res_info2):

    voltages        = np.zeros(shape=(N_voltages,3))
    voltages[:,0]   = amplitude*np.cos(freq*time_steps*1e8)
    
    target_electrode = len(topology_parameter["e_pos"]) - 1
    
    sim_class = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, folder=folder+"data/",
                                    seed=seed, add_to_path=f'_{np.round(freq,2)}_{R_val}_{seed}', res_info2=res_info2)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, stat_size=stat_size)

if __name__ == '__main__':

    # Parameter
    folder              = "scripts/2_funding_period/WP2/cos_input/resistance/"
    stat_size           = 1000
    eq_steps            = 1000000
    network_topology    = "cubic"
    seed                = 0
    N                   = 7

    # Network Topology
    topology_parameter  = {
        "Nx"    :   N,
        "Ny"    :   N,
        "Nz"    :   1,
        "e_pos" :   [[0,0,0],[N-1,N-1,0]],
    }

    # Time Scale
    step_size   = 1e-10
    N_voltages  = 10000
    time        = step_size*np.arange(N_voltages)
    amplitude   = 0.2

    for R_val in [50,100,200,400,800]:

        for seed in range(10):

            rs          = np.random.RandomState(seed=seed)
            res_info2   = {
                "R"         : R_val,
                "np_index"  : rs.choice(np.arange(1,48), 9, replace=False)
            }

            # Parameter
            frequencies = np.arange(0.1,2,0.2)
            N_processes = len(frequencies)

            procs = []
            for i in range(N_processes):

                process = multiprocessing.Process(target=run_simulation, args=(frequencies[i], time, network_topology, topology_parameter,
                                                                            eq_steps, folder, stat_size, seed, amplitude, N_voltages, R_val, res_info2))
                process.start()
                procs.append(process)
            
            for p in procs:
                p.join()

            frequencies = np.arange(2.1,4,0.2)

            procs = []
            for i in range(N_processes):

                process = multiprocessing.Process(target=run_simulation, args=(frequencies[i], time, network_topology, topology_parameter,
                                                                            eq_steps, folder, stat_size, seed, amplitude, N_voltages, R_val, res_info2))
                process.start()
                procs.append(process)
            
            for p in procs:
                p.join()