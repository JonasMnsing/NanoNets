"""
Apply a DC input at fixed voltage to the first node of a string of variable numbers of nanoparticles.
The system operates in a nonlinear regime as temperature is fixed at T=5K
The circuit is either closed (last node is connected to ground) or open (last node is floating)
"""

import sys
sys.path.append("src/")
import nanonets
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size):
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter,
                                              folder=folder, high_C_output=False)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode,
                               stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    # Parameter
    N_voltages  = 50000
    time_step   = 1e-10
    stat_size   = 200
    time_steps  = np.arange(N_voltages)*time_step
    # N_p_vals    = [2, 4, 6, 8, 10,12,14,16,18,20]
    # N_p_vals    = [22,24,26,28,30,32,34,36,38,40]
    N_p_vals    = [42,44,46,48,50,9]
    N_processes = len(N_p_vals)
    folder      = "/home/j/j_mens07/phd/data/2_funding_period/"
    # folder      = "/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/"

    # Voltages
    volt        = np.zeros(shape=(N_voltages,3))
    volt[:,0]   = 0.02

    # Closed Circuit
    # procs       = []
    # for i in range(N_processes):
    #     N_p                 = N_p_vals[i]
    #     topology_parameter  = {
    #         "Nx"                : N_p,
    #         "Ny"                : 1,
    #         "Nz"                : 1,
    #         "e_pos"             : [[0,0,0],[N_p-1,0,0]],
    #         "electrode_type"    : ['constant','constant']
    #     }
    #     process = multiprocessing.Process(target=run_simulation,
    #                                       args=(time_steps, volt, topology_parameter,
    #                                             folder+"current/wo_magic_cable/dc_input_vs_size/", stat_size))
    #     process.start()
    #     procs.append(process)
    # for p in procs:
    #     p.join()

    # Open Circuit
    procs       = []
    for i in range(N_processes):
        N_p                 = N_p_vals[i]
        topology_parameter  = {
            "Nx"                : N_p,
            "Ny"                : 1,
            "Nz"                : 1,
            "e_pos"             : [[0,0,0],[N_p-1,0,0]],
            "electrode_type"    : ['constant','floating']
        }
        process = multiprocessing.Process(target=run_simulation,
                                          args=(time_steps, volt, topology_parameter,
                                                folder+"potential/wo_magic_cable/dc_input_vs_size/", stat_size))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()