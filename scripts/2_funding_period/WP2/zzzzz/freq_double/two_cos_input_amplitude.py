import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets
import multiprocessing

# Simulation Function
def run_simulation(freq1, freq2, time_steps, network_topology, topology_parameter, eq_steps, folder, stat_size, seed, amplitude1, amplitude2, N_voltages):

    voltages        = np.zeros(shape=(N_voltages,4))
    voltages[:,0]   = amplitude1*np.cos(freq1*time_steps*1e8)
    voltages[:,1]   = amplitude2*np.cos(freq2*time_steps*1e8)
    
    target_electrode = len(topology_parameter["e_pos"]) - 1
    
    sim_class = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, folder=folder+"data/", seed=seed, add_to_path=f'_{np.round(amplitude1,3)}_{np.round(amplitude2,3)}')
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, stat_size=stat_size)

if __name__ == '__main__':

    # Parameter
    folder              = "scripts/2_funding_period/WP2/freq_double/amplitude/"
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
        "e_pos" :   [[int((N-1)/2),0,0],[0,int((N-1)/2),0],[N-1,N-1,0]],
    }

    # Time Scale
    step_size   = 1e-10
    N_voltages  = 10000
    time        = step_size*np.arange(N_voltages)
    f1          = 1.5
    f2          = 4.5
    amplitudes  = [[0.2,0.4],[0.2,0.6],[0.4,0.2],[0.6,0.2]]

    # Parameter
    N_processes = len(amplitudes)

    procs = []
    for i in range(N_processes):

        process = multiprocessing.Process(target=run_simulation, args=(f1, f2, time, network_topology,
                                                                       topology_parameter, eq_steps, folder, stat_size, seed, amplitudes[i][0], amplitudes[i][1], N_voltages))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()
