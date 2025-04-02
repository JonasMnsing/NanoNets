import sys
sys.path.append("src/")
import nanonets
import nanonets_utils
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size, f0):

    np_info2    = {
        'np_index'      : [81], 
        'mean_radius'   : 5e3,
        'std_radius'    : 0.0
    }
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, high_C_output=True, add_to_path=f"_{f0}", np_info2=np_info2)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    # Global
    stat_size           = 50
    N_periods           = 20
    time_step           = 1e-9
    N_p                 = 9
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
    folder      = "/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/potential/magic_cable/ac_input_vs_freq/"
    # freq_vals   = [10e3,20e3,40e3,80e3,160e3,320e3,640e3,1280e3,2560e3,5120e3]
    # freq_vals   = [10000,20000,40000,80000,160000,320000,640000,1280000,2560000,5120000]
    freq_vals   = [60e3,120e3,240e3,480e3,960e3,1280e3,1920e3,2560e3,3840e3,5120e3]
    N_processes = len(freq_vals)
    procs       = []

    for i in range(N_processes):
        f0                  = int(freq_vals[i])
        T_sim               = N_periods/f0
        N_voltages          = int(T_sim/time_step)
        frequencies         = [f0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        amplitudes          = [0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        time_steps, volt    = nanonets_utils.sinusoidal_voltages(N_voltages, topology_parameter, amplitudes=amplitudes, frequencies=frequencies,
                                                                 time_step=time_step)
        print(volt.shape)
        process             = multiprocessing.Process(target=run_simulation, args=(time_steps, volt, topology_parameter, folder, stat_size, f0))
        process.start()
    for p in procs:
        p.join()