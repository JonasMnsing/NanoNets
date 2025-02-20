import sys
sys.path.append("src/")
import nanonets
import nanonets_utils
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, np_info2, path, radius=1e6, eq_steps=0, T_val=0, stat_size=50):

    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=path, np_info2=np_info2, add_to_path=f"_{radius}")
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
    stat_size   = 50
    time_step   = 1e-7
    procs       = []
    # path        = '/home/j/j_mens07/phd/data/2_funding_period/potential/magic_cable/magnitude/'
    path        = '/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/potential/magic_cable/direct_current/'

    # Voltage Paramter
    # N_voltages      = 1000
    # N_min, N_max    = 2, 12
    # # N_min, N_max    = 12, 22
    # N_vals          = np.arange(N_min, N_max)

    # for i in range(N_processes):

    #     N_p             = N_vals[i]
    #     string_topology = return_string_topology(N_p)
    #     time_steps      = np.arange(N_voltages)*time_step
    #     voltages        = np.zeros(shape=(N_voltages,3))
    #     voltages[:,0]   = np.linspace(0, 0.2, N_voltages, endpoint=False)
    #     np_info2        = {
    #         'np_index'      : [string_topology["Nx"]], 
    #         'mean_radius'   : radius,
    #         'std_radius'    : 0.0
    #     }
    #     process         = multiprocessing.Process(target=run_simulation, args=(time_steps, voltages, string_topology, np_info2,
    #                                                                             path, radius, 0, T_val, stat_size))
    #     process.start()
    #     procs.append(process)

    # for p in procs:
    #     p.join()

    print('-----')

    # Voltage Paramter
    N_voltages      = 1000
    N_processes     = 6
    N_vals          = [3,5,7,9,11,13]

    for i in range(N_processes):

        N_p                 = N_vals[i]
        network_topology    = return_network_topology(N_p)
        time_steps          = np.arange(N_voltages)*time_step
        voltages            = np.zeros(shape=(N_voltages,9))
        voltages[:,0]       = np.linspace(0, 0.2, N_voltages, endpoint=False)
        np_info2            = {
            'np_index'      : [int(network_topology["Nx"]*network_topology["Ny"])], 
            'mean_radius'   : radius,
            'std_radius'    : 0.0
        }
        process             = multiprocessing.Process(target=run_simulation, args=(time_steps, voltages, network_topology, np_info2,
                                                                                       path, radius, 0, T_val, stat_size))
        process.start()
        procs.append(process)

    for p in procs:
        p.join()