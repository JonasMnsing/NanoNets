import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
sys.path.append("/home/j/j_mens07/phd/NanoNets/src/")

import nanonets
import multiprocessing

# Simulation Function
def run_simulation(freq, time_steps, network_topology, topology_parameter, eq_steps, folder, stat_size, seed, amplitude, N_voltages):

    voltages            = np.zeros(shape=(N_voltages,9))
    voltages[:,0]       = amplitude*np.cos(freq*time_steps*1e8)
    target_electrode    = 6
    
    sim_class = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, folder=folder+"data/", seed=seed, add_to_path=f'_{np.round(freq,2)}')
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, stat_size=stat_size, output_potential=True)

if __name__ == '__main__':

    # Parameter
    folder              = "scripts/2_funding_period/WP2/cos_input/uniform/"
    stat_size           = 1000
    eq_steps            = 0
    network_topology    = "cubic"
    seed                = 0
    N_particles         = 7

    # Network Topology
    topology_parameter  = {
        "Nx"                :   N_particles,
        "Ny"                :   N_particles,
        "Nz"                :   1,
        "e_pos"             :  [[0,0,0],[int((N_particles-1)/2),0,0],[N_particles-1,0,0],[0,int((N_particles-1)/2),0],[0,N_particles-1,0],
                                [int((N_particles-1)/2),N_particles-1,0],[N_particles-1,int((N_particles-1)/2),0],[N_particles-1,N_particles-1,0]],
    "electrode_type"        : ['constant','floating','floating','floating','floating','floating','floating','constant']
    }

    # Time Scale
    step_size   = 1e-10
    N_voltages  = 10000
    time        = step_size*np.arange(N_voltages)
    amplitude   = 0.2

    # Parameter
    frequencies = np.arange(0.1,2,0.2)
    N_processes = len(frequencies)

    procs = []
    for i in range(N_processes):

        process = multiprocessing.Process(target=run_simulation, args=(frequencies[i], time, network_topology, topology_parameter,
                                                                       eq_steps, folder, stat_size, seed, amplitude, N_voltages))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()

    frequencies = np.arange(0.2,2.1,0.2)

    procs = []
    for i in range(N_processes):

        process = multiprocessing.Process(target=run_simulation, args=(frequencies[i], time, network_topology, topology_parameter,
                                                                       eq_steps, folder, stat_size, seed, amplitude, N_voltages))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()

    frequencies = np.arange(2.1,4,0.2)

    procs = []
    for i in range(N_processes):

        process = multiprocessing.Process(target=run_simulation, args=(frequencies[i], time, network_topology, topology_parameter,
                                                                       eq_steps, folder, stat_size, seed, amplitude, N_voltages))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()

    frequencies = np.arange(2.2,4.1,0.2)

    procs = []
    for i in range(N_processes):

        process = multiprocessing.Process(target=run_simulation, args=(frequencies[i], time, network_topology, topology_parameter,
                                                                       eq_steps, folder, stat_size, seed, amplitude, N_voltages))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()

    frequencies = np.arange(4.1,6,0.2)

    procs = []
    for i in range(N_processes):

        process = multiprocessing.Process(target=run_simulation, args=(frequencies[i], time, network_topology, topology_parameter,
                                                                       eq_steps, folder, stat_size, seed, amplitude, N_voltages))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()

    procs = []
    for i in range(N_processes):

        process = multiprocessing.Process(target=run_simulation, args=(frequencies[i], time, network_topology, topology_parameter,
                                                                       eq_steps, folder, stat_size, seed, amplitude, N_voltages))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()

    frequencies = np.arange(4.2,6.1,0.2)

    procs = []
    for i in range(N_processes):

        process = multiprocessing.Process(target=run_simulation, args=(frequencies[i], time, network_topology, topology_parameter,
                                                                       eq_steps, folder, stat_size, seed, amplitude, N_voltages))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()