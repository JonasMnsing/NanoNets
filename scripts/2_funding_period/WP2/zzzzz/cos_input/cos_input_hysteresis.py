import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

import nanonets
import multiprocessing

# Simulation Function
def run_simulation(voltages, time_steps, network_topology, topology_parameter, eq_steps, folder, stat_size, res_info2=None, np_info2=None):
    
    target_electrode = len(topology_parameter["e_pos"]) - 1
    
    sim_class = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, folder=folder, res_info2=res_info2, np_info2=np_info2)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, stat_size=stat_size)

if __name__ == '__main__':

    # Parameter
    folder              = "scripts/2_funding_period/WP2/cos_input/hysteresis/data/"
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
    n_vals      = 633
    N_voltages  = 20*n_vals
    time        = step_size*np.arange(N_voltages)
    amplitude   = 0.2
    values      = []
    time_vals   = []
    last_time   = 0

    for i, freq in enumerate(np.arange(1,2.1,0.1)):

        values.extend(list(amplitude*np.cos(freq*time[0:int(n_vals/freq)]*1e8)))
        time_vals.extend(list(np.arange(last_time, last_time+int(n_vals/freq))))
        last_time = int(n_vals/freq)

    for i, freq in enumerate(np.arange(2,0.9,-0.1)):

        values.extend(list(amplitude*np.cos(freq*time[0:int(n_vals/freq)]*1e8)))
        time_vals.extend(list(np.arange(last_time, last_time+int(n_vals/freq))))
        last_time = int(n_vals/freq)

    voltages        = np.zeros(shape=(len(values),3))
    voltages[:,0]   = values

    # Uniform
    run_simulation(voltages, time, network_topology, topology_parameter, eq_steps, folder, stat_size)

    seed    = 8
    rs      = np.random.RandomState(seed=seed)

    res_info2   = {
        "R"         : 200,
        "np_index"  : rs.choice(np.arange(1,48), 9, replace=False)
    }
    folder  = "scripts/2_funding_period/WP2/cos_input/hysteresis/data_R/"

    # Resistance
    run_simulation(voltages, time, network_topology, topology_parameter, eq_steps, folder, stat_size, res_info2=res_info2)

    np_info2   = {
        "mean_radius"   : 40,
        "std_radius"    : 0.0,
        "np_index"      : rs.choice(np.arange(1,48), 9, replace=False)
    }
    folder  = "scripts/2_funding_period/WP2/cos_input/hysteresis/data_rad/"

    # Radius
    run_simulation(voltages, time, network_topology, topology_parameter, eq_steps, folder, stat_size, np_info2=np_info2)
