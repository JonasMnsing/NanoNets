"""
Run KMC Code for const set of voltages with fixed parameters defined in "params.csv" @ folder.
Output Electrode @ last position in topology_parameter key "pos"
"""

import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
sys.path.append("/home/j/j_mens07/phd/NanoNets/src/")

import nanonets
import nanonets_utils
import multiprocessing

# Simulation Function
def run_simulation(voltages, time_steps, network_topology, topology_parameter, np_info2, eq_steps, folder, stat_size, seed):

    target_electrode    = len(topology_parameter["e_pos"]) - 1
    r_val               = np_info2["mean_radius"]
    
    sim_class = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, np_info2=np_info2, folder=folder+"data/", seed=seed, add_to_path=f'_{r_val}_{seed}')
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, stat_size=stat_size, output_potential=True)
        

if __name__ == '__main__':

    # Time Scale
    step_size   = 1e-10
    N_voltages  = 4000
    time_steps  = step_size*np.arange(N_voltages)

    # Voltages
    off_state   = 0.1
    on_state    = 0.2
    on_t1       = 200
    on_t2       = 400
    voltages    = np.zeros(shape=(N_voltages,9))

    # Input Electrode
    voltages[:,0]           = np.repeat(off_state, N_voltages)
    voltages[on_t1:on_t2,0] = on_state

    # Parameter
    folder              = "scripts/2_funding_period/WP2/step_input/1I_1O_radius/"
    stat_size           = 500
    eq_steps            = 10000
    network_topology    = "cubic"
    N                   = 7

    # Network Topology
    topology_parameter  = {
        "Nx"                :   N,
        "Ny"                :   N,
        "Nz"                :   1,
        "e_pos"             :  [[0,0,0],[int((N-1)/2),0,0],[N-1,0,0],
                                [0,int((N-1)/2),0],[0,N-1,0],[int((N-1)/2),N-1,0],
                                [N-1,int((N-1)/2),0],[N-1,N-1,0]],
        "electrode_type"    : ['constant','floating','floating','floating','floating','floating','floating','floating']
    }

    for r_val in [5,15,20,25,30,35,40]:

        procs = []
        for seed in range(10):

            rs          = np.random.RandomState(seed=seed)
            np_info2    = {
                "mean_radius"   : r_val,
                "std_radius"    : 0.0,
                "np_index"      : rs.choice(np.arange(1,48), 9, replace=False)
            }

            process = multiprocessing.Process(target=run_simulation, args=(voltages, time_steps, network_topology, topology_parameter, np_info2, eq_steps, folder, stat_size, seed))
            process.start()
            procs.append(process)
        
        for p in procs:
            p.join()