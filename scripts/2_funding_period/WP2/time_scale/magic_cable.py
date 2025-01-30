import sys
sys.path.append("../../../../../src")
import nanonets
import numpy as np
import multiprocessing

def run_simulation(thread, time_steps, voltages, radius, N_p, folder, stat_size):
    
    topology_parameter  = {
        "Nx"                : N_p,
        "Ny"                : N_p,
        "Nz"                : 1,
        "e_pos"             : [[(N_p-1)//2,0,0],[0,0,0],[N_p-1,0,0],
                               [0,(N_p-1)//2,0],[N_p-1,(N_p-1)//2,0],
                               [0,N_p-1,0],[N_p-1,N_p-1,0],[(N_p-1)//2,N_p-1,0]],
        "electrode_type"    : ['constant','constant','constant',
                               'constant','constant','constant',
                               'constant','floating']
    }

    np_info2 = {
        'np_index'      : [topology_parameter["Nx"]**2], 
        'mean_radius'   : radius,
        'std_radius'    : 0.0
    }

    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, np_info2=np_info2, add_to_path=f'_{thread}')
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, stat_size=stat_size, save=True)

if __name__ == '__main__':

    N_voltages          = 10000
    time_step           = 1e-7
    time_steps          = np.arange(N_voltages)*time_step
    voltages            = np.zeros(shape=(N_voltages,9))
    voltages[:1000,0]   = 0.1

    folder      = "/scratch/j_mens07/data/2_funding_period/potential/time_scale/"
    radius_vals = [1e1,1e2,1e3,1e4,1e5,1e6,1e7]
    N_p_vals    = [3,5,7,9,11]
    N_processes = len(radius_vals)
    stat_size   = 10

    for i in range(N_processes):
        radius  = radius_vals[i]
        for N_p in N_p_vals:
            process = multiprocessing.Process(target=run_simulation, args=(i, time_steps, voltages, radius, N_p, folder, stat_size))
            process.start()