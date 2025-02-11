import sys
sys.path.append("src/")
import nanonets
import numpy as np
import multiprocessing

def run_simulation(thread, time_steps, voltages, N_p, folder, stat_size):
    
    topology_parameter  = {
        "Nx"                : N_p,
        "Ny"                : N_p,
        "Nz"                : 1,
        "e_pos"             : [[(N_p-1)//2,0,0],[0,0,0],[N_p-1,0,0],
                               [0,(N_p-1)//2,0],[N_p-1,(N_p-1)//2,0],
                               [0,N_p-1,0],[N_p-1,N_p-1,0],[(N_p-1)//2,N_p-1,0]],
        "electrode_type"    : ['constant','constant','constant','constant',
                               'constant','constant','constant','floating']
    }

    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, stat_size=stat_size, save=True)

if __name__ == '__main__':

    N_voltages          = 4000
    time_step           = 1e-10
    time_steps          = np.arange(N_voltages)*time_step
    voltages            = np.zeros(shape=(N_voltages,9))
    voltages[:1500,0]   = 0.1

    # folder      = "/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/potential/wo_magic_cable/time_scale/"
    folder      = "/home/j/j_mens07/phd/data/2_funding_period/potential/wo_magic_cable/time_scale/"
    # N_p_vals    = [5,7,9,11]
    N_p_vals    = [3,13]
    N_processes = 2
    stat_size   = 500

    for i in range(N_processes):
        N_p     = N_p_vals[i]
        process = multiprocessing.Process(target=run_simulation, args=(i, time_steps, voltages, N_p, folder, stat_size))
        process.start()