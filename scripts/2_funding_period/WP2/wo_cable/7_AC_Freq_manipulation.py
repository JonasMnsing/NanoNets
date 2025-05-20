import sys
sys.path.append("src/")
import nanonets
import nanonets_utils
import numpy as np
import multiprocessing

def run_simulation(p, time_steps, voltages, volt_controls, topology_parameter, folder, stat_size):
    
    target_electrode    = len(topology_parameter["e_pos"])-1

    for i, volt_c in enumerate(volt_controls):
        volt            = voltages.copy()
        volt[:,0]       = volt_c[0]
        volt[:,3:-2]    = volt_c[1:]
        sim_class       = nanonets.simulation(topology_parameter=topology_parameter, folder=folder,
                                              high_C_output=False, add_to_path=f"_{int(p*50+i)}")
        sim_class.run_var_voltages(voltages=volt, time_steps=time_steps, target_electrode=target_electrode,
                                   stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    # Global
    stat_size   = 50
    N_periods   = 40
    N_p         = 9
    folder      = "/home/j/j_mens07/phd/data/2_funding_period/"
    f0          = 1.0*1e6
    f1          = 3.0*1e6
    dt          = 1/(20 * f1)
    T_sim       = N_periods/f1
    N_voltages  = int(T_sim/dt)

    # 8 Electrodes
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

    # Frequency Points
    N_samples   = 500
    N_processes = 10
    volt_sample = np.random.uniform(-0.05,0.05,(N_samples,5))

    frequencies         = [0.0,f0,f1,0.0,0.0,0.0,0.0,0.0]
    amplitudes          = [0.0,0.1,0.1,0.0,0.0,0.0,0.0,0.0]
    offsets             = []
    time_steps, volt    = nanonets_utils.sinusoidal_voltages(N_voltages, topology_parameter,
                                                                amplitudes=amplitudes,
                                                                frequencies=frequencies,
                                                                time_step=dt)

    procs   = []
    for i in range(N_processes):
        volt_controls   = nanonets_utils.distribute_array_across_processes(i, volt_sample, N_processes)
        process         = multiprocessing.Process(target=run_simulation, args=(i, time_steps, volt, volt_controls, topology_parameter,
                                                                               folder+"potential/wo_magic_cable/ac_two_tone_signal/", stat_size))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()