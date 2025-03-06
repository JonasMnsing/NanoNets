import sys
sys.path.append("src/")
import nanonets
import nanonets_utils
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size, f0):
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, high_C_output=False, add_to_path=f"_{f0}")
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    # Global
    N_voltages  = 200000
    time_step   = 1e-10
    stat_size   = 100
    time_steps  = np.arange(N_voltages)*time_step
    folder      = "/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/potential/wo_magic_cable/frequency/"
    # folder      = "/home/j/j_mens07/phd/data/2_funding_period/potential/wo_magic_cable/frequency/"
    
    topology_parameter  = {
        "Nx"                : 10,
        "Ny"                : 1,
        "Nz"                : 1,
        "e_pos"             : [[0,0,0],[9,0,0]],
        "electrode_type"    : ['constant','floating']
    }

    # String
    freq_vals   = np.linspace(0.5,10,20)
    N_processes = len(freq_vals)
    procs       = []

    for i in range(N_processes):
        f0                  = freq_vals[i]
        frequencies         = [f0*1e6,0.0]
        amplitudes          = [0.1,0.0]
        time_steps, volt    = nanonets_utils.sinusoidal_voltages(N_voltages, topology_parameter, amplitudes=amplitudes, frequencies=frequencies, time_step=time_step)
        process = multiprocessing.Process(target=run_simulation, args=(time_steps, volt, topology_parameter, folder, stat_size, f0))
        process.start()
        procs.append(process)
    for p in procs:
            p.join()

    # Network    
    # N_p_vals    = [3,5,7,9,11,13]
    # N_processes = len(N_p_vals)
    # procs       = []
    # volt        = np.zeros(shape=(N_voltages,9))
    # volt[:,0]   = 0.1
    # for i in range(N_processes):
    #     N_p                 = N_p_vals[i]
    #     topology_parameter  = {
    #         "Nx"                : N_p,
    #         "Ny"                : N_p,
    #         "Nz"                : 1,
    #         "e_pos"             : [[(N_p-1)//2,0,0],[0,0,0],[N_p-1,0,0],
    #                             [0,(N_p-1)//2,0],[N_p-1,(N_p-1)//2,0],
    #                             [0,N_p-1,0],[N_p-1,N_p-1,0],[(N_p-1)//2,N_p-1,0]],
    #         "electrode_type"    : ['constant','constant','constant','constant',
    #                             'constant','constant','constant','floating']
    #     }
    #     process = multiprocessing.Process(target=run_simulation, args=(time_steps, volt, topology_parameter, folder, stat_size))
    #     process.start()
    # for p in procs:
    #         p.join()