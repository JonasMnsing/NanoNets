import sys
sys.path.append("src/")
import nanonets
import nanonets_utils
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, np_info2, path, f, eq_steps=0, T_val=0, stat_size=50):

    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=path, np_info2=np_info2, add_to_path=f"_{f}")
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, T_val=T_val,
                               stat_size=stat_size, save=True, eq_steps=eq_steps)
    
def return_string_topology(N_p: int, floating_output: bool = True) -> dict:

    if floating_output:
        topology_parameter  = {
            "Nx"                : N_p,
            "Ny"                : 1,
            "Nz"                : 1,
            "e_pos"             : [[0,0,0],[N_p-1,0,0]],
            "electrode_type"    : ['constant','floating']
        }
    else:
        topology_parameter  = {
            "Nx"                : N_p,
            "Ny"                : 1,
            "Nz"                : 1,
            "e_pos"             : [[0,0,0],[N_p-1,0,0]],
            "electrode_type"    : ['constant','constant']
        }

    return topology_parameter

def return_network_topology(N_p: int, floating_output: bool = True) -> dict:

    if floating_output:
        topology_parameter  = {
            "Nx"                : N_p,
            "Ny"                : N_p,
            "Nz"                : 1,
            "e_pos"             : [[(N_p-1)//2,0,0],[0,0,0],[N_p-1,0,0],[0,(N_p-1)//2,0],
                                [N_p-1,(N_p-1)//2,0],[0,N_p-1,0],[N_p-1,N_p-1,0],[(N_p-1)//2,N_p-1,0]],
            "electrode_type"    : ['constant','constant','constant','constant','constant','constant','constant','floating']
        }
    else:
        topology_parameter  = {
            "Nx"                : N_p,
            "Ny"                : N_p,
            "Nz"                : 1,
            "e_pos"             : [[(N_p-1)//2,0,0],[0,0,0],[N_p-1,0,0],[0,(N_p-1)//2,0],
                                [N_p-1,(N_p-1)//2,0],[0,N_p-1,0],[N_p-1,N_p-1,0],[(N_p-1)//2,N_p-1,0]],
            "electrode_type"    : ['constant','constant','constant','constant','constant','constant','constant','constant']
        }

    return topology_parameter

if __name__ == '__main__':

    N_processes = 10
    radius      = 1e6
    T_val       = 5.0
    procs       = []
    path        = '/home/j/j_mens07/phd/data/2_funding_period/potential/magic_cable/freq/'
    # path        = '/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/potential/magic_cable/system_size/'

    # Voltage Paramter
    N_voltages      = 5000
    amplitudes      = [0.1,0.0]
    f_vals          = [1e8,1e7,1e6]
    step_vals       = [1e-10,1e-10,1e-10]
    stat_vals       = [500,500,500]
    N_processes     = len(f_vals)

    for i in range(N_processes):

        frequencies             = [f_vals[i],0.0]
        time_step               = step_vals[i]
        stat_size               = stat_vals[i]
        string_topology         = return_string_topology(11)
        time_steps, voltages    = nanonets_utils.sinusoidal_voltages(N_samples=N_voltages, topology_parameter=string_topology,
                                                                    amplitudes=amplitudes, frequencies=frequencies, time_step=time_step)
        np_info2                = {
            'np_index'      : [string_topology["Nx"]], 
            'mean_radius'   : radius,
            'std_radius'    : 0.0
        }
        process                 = multiprocessing.Process(target=run_simulation, args=(time_steps, voltages, string_topology, np_info2,
                                                                                       path, f_vals[i], 0, T_val, stat_size))
        process.start()
        procs.append(process)

    for p in procs:
        p.join()

    # Voltage Paramter
    N_voltages      = 1000
    amplitudes      = [0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    for i in range(N_processes):

        frequencies             = [f_vals[i],0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        time_step               = step_vals
        stat_size               = stat_vals[i]
        network_topology        = return_network_topology(11)
        time_steps, voltages    = nanonets_utils.sinusoidal_voltages(N_samples=N_voltages, topology_parameter=network_topology,
                                                                    amplitudes=amplitudes, frequencies=frequencies, time_step=time_step)
        np_info2                = {
            'np_index'      : [int(network_topology["Nx"]**2)], 
            'mean_radius'   : radius,
            'std_radius'    : 0.0
        }
        process                 = multiprocessing.Process(target=run_simulation, args=(time_steps, voltages, network_topology, np_info2,
                                                                                       path, f_vals[i], 0, T_val, stat_size))
        process.start()
        procs.append(process)

    for p in procs:
        p.join()