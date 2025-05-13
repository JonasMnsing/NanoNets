import sys
sys.path.append("src/")
import nanonets
import nanonets_utils
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size, f0):

    np_info2    = {
        'np_index'      : [81], 
        'mean_radius'   : 5e4,
        'std_radius'    : 0.0
    }
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder,
                                              high_C_output=True, add_to_path=f"_{f0/1e6}", np_info2=np_info2)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode,
                               stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    # Global
    base_stat_size  = 20
    N_periods       = 40
    N_p             = 9
    folder          = "/home/j/j_mens07/phd/data/2_funding_period/"

    # Frequency Points
    freq_vals   = [18.,23.,28.,36.,44.,55.,68.,86.,105.,133.]
    N_processes = len(freq_vals)
    stat_size_v = np.round(300*np.array(freq_vals)/5.0)

    # 2 Electrodes
    procs   = []
    topology_parameter  = {
        "Nx"                : N_p,
        "Ny"                : N_p,
        "Nz"                : 1,
        "e_pos"             : [[(N_p-1)//2,0,0],[(N_p-1)//2,N_p-1,0]],
        "electrode_type"    : ['constant','floating']
    }

    for i in range(N_processes):
        f0                  = freq_vals[i]*1e6      # Convert MHz to Hz
        dt                  = 1/(40 * f0)           # 20 Samples per period
        T_sim               = N_periods/f0
        N_voltages          = int(T_sim/dt)
        frequencies         = [f0,0.0]
        amplitudes          = [0.1,0.0]
        stat_size           = max(base_stat_size, int(stat_size_v[i]))
        time_steps, volt    = nanonets_utils.sinusoidal_voltages(
            N_voltages, topology_parameter,amplitudes=amplitudes,
            frequencies=frequencies,time_step=dt)
        
        process = multiprocessing.Process(
            target=run_simulation,
            args=(time_steps, volt, topology_parameter,
                  folder+"potential/magic_cable/ac_input_vs_freq/",
                  stat_size, f0))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()

    # Frequency Points
    freq_vals   = [0.25,0.5,1.,2.,5.,6.,8.,10.,12.,15.]
    N_processes = len(freq_vals)
    stat_size_v = np.round(300*np.array(freq_vals)/5.0)

    # 2 Electrodes
    procs   = []
    topology_parameter  = {
        "Nx"                : N_p,
        "Ny"                : N_p,
        "Nz"                : 1,
        "e_pos"             : [[(N_p-1)//2,0,0],[(N_p-1)//2,N_p-1,0]],
        "electrode_type"    : ['constant','floating']
    }

    for i in range(N_processes):
        f0                  = freq_vals[i]*1e6      # Convert MHz to Hz
        dt                  = 1/(40 * f0)           # 20 Samples per period
        T_sim               = N_periods/f0
        N_voltages          = int(T_sim/dt)
        frequencies         = [f0,0.0]
        amplitudes          = [0.1,0.0]
        stat_size           = max(base_stat_size, int(stat_size_v[i]))
        time_steps, volt    = nanonets_utils.sinusoidal_voltages(
            N_voltages, topology_parameter,amplitudes=amplitudes,
            frequencies=frequencies,time_step=dt)
        
        process = multiprocessing.Process(
            target=run_simulation,
            args=(time_steps, volt, topology_parameter,
                  folder+"potential/magic_cable/ac_input_vs_freq/",
                  stat_size, f0))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()

    # Frequency Points
    freq_vals   = [0.001,0.002,0.004,0.006,0.008,0.01,0.03,0.06,0.12]
    N_processes = len(freq_vals)
    stat_size_v = np.round(300*np.array(freq_vals)/5.0)

    # 2 Electrodes
    procs   = []
    topology_parameter  = {
        "Nx"                : N_p,
        "Ny"                : N_p,
        "Nz"                : 1,
        "e_pos"             : [[(N_p-1)//2,0,0],[(N_p-1)//2,N_p-1,0]],
        "electrode_type"    : ['constant','floating']
    }

    for i in range(N_processes):
        f0                  = freq_vals[i]*1e6      # Convert MHz to Hz
        dt                  = 1/(40 * f0)           # 20 Samples per period
        T_sim               = N_periods/f0
        N_voltages          = int(T_sim/dt)
        frequencies         = [f0,0.0]
        amplitudes          = [0.1,0.0]
        stat_size           = max(base_stat_size, int(stat_size_v[i]))
        time_steps, volt    = nanonets_utils.sinusoidal_voltages(
            N_voltages, topology_parameter,amplitudes=amplitudes,
            frequencies=frequencies,time_step=dt)
        
        process = multiprocessing.Process(
            target=run_simulation,
            args=(time_steps, volt, topology_parameter,
                  folder+"potential/magic_cable/ac_input_vs_freq/",
                  stat_size, f0))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()

    # # 8 Electrodes
    # procs               = []
    # topology_parameter  = {
    #     "Nx"                : N_p,
    #     "Ny"                : N_p,
    #     "Nz"                : 1,
    #     "e_pos"             : [[(N_p-1)//2,0,0],[0,0,0],[N_p-1,0,0],
    #                         [0,(N_p-1)//2,0],[N_p-1,(N_p-1)//2,0],
    #                         [0,N_p-1,0],[N_p-1,N_p-1,0],[(N_p-1)//2,N_p-1,0]],
    #     "electrode_type"    : ['constant','constant','constant','constant',
    #                         'constant','constant','constant','floating']
    # }

    # for i in range(N_processes):
    #     f0                  = freq_vals[i]*1e6      # Convert MHz to Hz
    #     dt                  = 1/(20 * f0)           # 20 Samples per period
    #     T_sim               = N_periods/f0
    #     N_voltages          = int(T_sim/dt)
    #     frequencies         = [f0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    #     amplitudes          = [0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    #     stat_size           = max(base_stat_size, int(stat_size_v[i]))
    #     time_steps, volt    = nanonets_utils.sinusoidal_voltages(N_voltages, topology_parameter,
    #                                                              amplitudes=amplitudes,
    #                                                              frequencies=frequencies,
    #                                                              time_step=dt)
    #     process = multiprocessing.Process(target=run_simulation,
    #                                       args=(time_steps, volt, topology_parameter,
    #                                             folder+"potential/magic_cable/ac_input_vs_freq/",
    #                                             stat_size, f0))
    #     process.start()
    #     procs.append(process)
    # for p in procs:
    #     p.join()