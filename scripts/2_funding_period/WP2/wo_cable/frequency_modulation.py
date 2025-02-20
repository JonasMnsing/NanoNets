import sys
sys.path.append("src/")
import nanonets
import nanonets_utils
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size, run):
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, high_C_output=False, add_to_path=f"_{run}")
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    # Global
    N_voltages  = 50000
    N_samples   = 500
    time_step   = 1e-10
    stat_size   = 100
    time_steps  = np.arange(N_voltages)*time_step
    folder      = "/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/potential/wo_magic_cable/frequency_modulation/"
    # folder      = "/home/j/j_mens07/phd/data/2_funding_period/potential/wo_magic_cable/frequency_modulation/"
    
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

    # String
    frequencies     = [3.5*1e6,0,0,0,0,0,0,0]
    amplitudes      = [0.1,0,0,0,0,0,0,0]
    offsets         = np.round(np.random.uniform(-0.1,0.1,(N_samples,8)),4)
    offsets[:,0]    = 0.0
    offsets[:,-1]   = 0.0
    N_processes     = 10
    procs           = []

    for n in range(0,N_samples,10):
        for i in range(N_processes):
            run                 = n+i
            offset              = offsets[run,:]
            time_steps, volt    = nanonets_utils.sinusoidal_voltages(N_voltages, topology_parameter, amplitudes=amplitudes, frequencies=frequencies, time_step=time_step, offset=offset)
            process = multiprocessing.Process(target=run_simulation, args=(time_steps, volt, topology_parameter, folder, stat_size, run))
            process.start()
            procs.append(process)
        for p in procs:
            p.join()