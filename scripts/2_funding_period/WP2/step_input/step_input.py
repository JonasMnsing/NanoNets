import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
sys.path.append("/home/j/j_mens07/phd/NanoNets/src/")

import nanonets

# Simulation Function
def run_simulation(voltages, time_steps, network_topology, topology_parameter, eq_steps, folder, stat_size, seed):

    target_electrode = len(topology_parameter["e_pos"]) - 1
    
    sim_class = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, folder=folder+"data/", seed=seed)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, stat_size=stat_size, output_potential=True)

if __name__ == '__main__':

    # Time Scale
    step_size   = 1e-10
    N_voltages  = 2000
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
    folder              = "scripts/2_funding_period/WP2/step_input/1I_1O/"
    stat_size           = 500
    eq_steps            = 0
    network_topology    = "cubic"
    seed                = 0
    # scale               = np.array([0.75, 0.82, 0.89, 0.96, 1.  , 1.06, 1.11, 1.17, 1.22, 1.28, 1.33, 1.41, 1.5 , 1.62, 1.85, 2.13])

    for i,N in enumerate(range(3,13)):

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

        run_simulation(voltages, time_steps, network_topology, topology_parameter, eq_steps, folder, stat_size, seed)
