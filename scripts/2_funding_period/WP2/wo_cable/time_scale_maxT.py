import sys
sys.path.append("src/")
import nanonets
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size):
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode,
                               stat_size=stat_size, save=True, T_val=293.0)

if __name__ == '__main__':

    # Global
    N_voltages  = 5000
    time_step   = 1e-10
    stat_size   = 500
    time_steps  = np.arange(N_voltages)*time_step
    # folder      = "/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/potential/wo_magic_cable/time_scale/maxT/"
    folder      = "/home/j/j_mens07/phd/data/2_funding_period/potential/wo_magic_cable/time_scale/maxT/"
    
    # String
    N_p_vals    = np.arange(2,13)
    N_processes = len(N_p_vals)
    procs       = []
    volt        = np.zeros(shape=(N_voltages,3))
    volt[:,0]   = 0.1
    for i in range(N_processes):
        N_p                 = N_p_vals[i]
        topology_parameter  = {
            "Nx"                : N_p,
            "Ny"                : 1,
            "Nz"                : 1,
            "e_pos"             : [[0,0,0],[N_p-1,0,0]],
            "electrode_type"    : ['constant','floating']
        }
        process = multiprocessing.Process(target=run_simulation, args=(time_steps, volt, topology_parameter, folder, stat_size))
        process.start()
    for p in procs:
            p.join()