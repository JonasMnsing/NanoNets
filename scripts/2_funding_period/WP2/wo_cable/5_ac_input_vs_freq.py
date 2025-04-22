import sys
sys.path.append("src/")
import nanonets
import nanonets_utils
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size, f0):
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder,
                                              high_C_output=False, add_to_path=f"_{f0}")
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode,
                               stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    # Global
    stat_size   = 50
    N_periods   = 20
    time_step   = 1e-9
    N_np        = 50
    freq_vals   = [0.005,0.01,0.05,0.1,0.5,1.0,5.0,10.0]
    N_processes = len(freq_vals)
    folder      = "/home/j/j_mens07/phd/data/2_funding_period/"
    # folder      = "/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/"

    # Closed Circuit
    procs               = []
    topology_parameter  = {
        "Nx"                : N_np,
        "Ny"                : 1,
        "Nz"                : 1,
        "e_pos"             : [[0,0,0],[N_np-1,0,0]],
        "electrode_type"    : ['constant','constant']
    }

    for i in range(N_processes):
        f0                  = freq_vals[i]*1e6
        T_sim               = N_periods/f0
        N_voltages          = int(T_sim/time_step)
        frequencies         = [f0,0.0]
        amplitudes          = [0.1,0.0]
        time_steps, volt    = nanonets_utils.sinusoidal_voltages(N_voltages, topology_parameter,
                                                                 amplitudes=amplitudes,
                                                                 frequencies=frequencies,
                                                                 time_step=time_step)
        process = multiprocessing.Process(target=run_simulation,
                                          args=(time_steps, volt, topology_parameter,
                                                folder+"current/wo_magic_cable/ac_input_vs_freq/",
                                                stat_size, f0))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()

    # Open Circuit
    procs               = []
    topology_parameter  = {
        "Nx"                : N_np,
        "Ny"                : 1,
        "Nz"                : 1,
        "e_pos"             : [[0,0,0],[N_np-1,0,0]],
        "electrode_type"    : ['constant','floating']
    }

    for i in range(N_processes):
        f0                  = freq_vals[i]*1e6
        T_sim               = N_periods/f0
        N_voltages          = int(T_sim/time_step)
        frequencies         = [f0,0.0]
        amplitudes          = [0.1,0.0]
        time_steps, volt    = nanonets_utils.sinusoidal_voltages(N_voltages, topology_parameter,
                                                                 amplitudes=amplitudes,
                                                                 frequencies=frequencies,
                                                                 time_step=time_step)
        process = multiprocessing.Process(target=run_simulation,
                                          args=(time_steps, volt, topology_parameter,
                                                folder+"potential/wo_magic_cable/ac_input_vs_freq/",
                                                stat_size, f0))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()