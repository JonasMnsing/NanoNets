import sys
sys.path.append("src/")
import nanonets
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size, sample):

    np_info2    = {
        'np_index'      : [81], 
        'mean_radius'   : 5e3,
        'std_radius'    : 0.0
    }
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, high_C_output=True, np_info2=np_info2, add_to_path=f'_{sample}')
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    folder              = "/home/j/j_mens07/phd/data/2_funding_period/potential/magic_cable/two_tone_signal/"
    topology_parameter  = {
        "Nx"                : 9,
        "Ny"                : 9,
        "Nz"                : 1,
        "e_pos"             : [[4,0,0],[0,0,0],[8,0,0],[0,4,0],
                               [8,4,0],[0,8,0],[8,8,0],[4,8,0]],
        "electrode_type"    : ['constant','constant','constant','constant',
                               'constant','constant','constant','floating']
    }
    
    # Voltages
    N_voltages  = 1000000
    N_samples   = 1008
    N_processes = 36
    time_step   = 1e-9
    U_0         = 0.1
    U_C         = 0.05
    time_steps  = np.arange(N_voltages)*time_step
    f0          = 40e3
    f1          = 140e3
    stat_size   = 10
    cap         = 5e3
    U_i         = U_0*np.sin(2*np.pi*f0*time_steps) + U_0*np.sin(2*np.pi*f1*time_steps)
    volt_C      = np.random.uniform(-U_C, U_C, (N_samples,6))
    procs       = []

    for i in range(N_samples):
        volt        = np.zeros(shape=(N_voltages,9))
        volt[:,0]   = U_i
        volt[:,1:7] = volt_C[i,:]
        volt        = np.round(volt,4)
    
        process = multiprocessing.Process(target=run_simulation, args=(time_steps, volt, topology_parameter, folder, stat_size, i))
        process.start()
        procs.append(process)

        if len(procs) >= N_processes:
            for p in procs:
                p.join()
            procs = [] 