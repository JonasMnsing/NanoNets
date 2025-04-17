import sys
sys.path.append("src/")
import nanonets
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size, steps_between):
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, add_to_path=f'_{steps_between}')
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    N_voltages          = 10000
    time_step           = 1e-10
    stat_size           = 500
    time_steps          = np.arange(N_voltages)*time_step
    N_p                 = 9
    topology_parameter  = {
        "Nx"                : N_p,
        "Ny"                : N_p,
        "Nz"                : 1,
        "e_pos"             : [[(N_p-1)//2,0,0],[0,0,0],[N_p-1,0,0],
                            [0,(N_p-1)//2,0],[N_p-1,(N_p-1)//2,0],
                            [0,N_p-1,0],[N_p-1,N_p-1,0],[(N_p-1)//2,N_p-1,0]],
        "electrode_type"    : ['constant','constant','constant','constant',
                            'constant','constant','constant','constant']
    }
    if topology_parameter["electrode_type"][-1] == 'floating':
        # folder      = "/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/potential/wo_magic_cable/dc_two_step_input/"
        folder      = "/home/j/j_mens07/phd/data/2_funding_period/potential/wo_magic_cable/dc_two_step_input/"
    else:
        # folder      = "/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/current/wo_magic_cable/dc_two_step_input/"
        folder      = "/home/j/j_mens07/phd/data/2_funding_period/current/wo_magic_cable/dc_two_step_input/"

    U_0             = 0.1
    relaxation_time = 30e-9
    steps_per_step  = int(np.round(relaxation_time/time_step))
    time_between    = [0,1e-9,2e-9,4e-9,8e-9,16e-9,32e-9,64e-9,128e-9,256e-9]
    # time_between    = [512e-9,1024e-9,2048e-9,4096e-9,8192e-9]
    steps_between   = [int(np.round(t/time_step)) for t in time_between]
    N_processes     = len(steps_between)
    procs           = []
    
    for i in range(N_processes):
        s                                           = steps_between[i]
        volt                                        = np.zeros((N_voltages,9))
        volt[:steps_per_step,0]                     = U_0
        volt[steps_per_step+s:2*steps_per_step+s,0] = U_0

        process = multiprocessing.Process(target=run_simulation, args=(time_steps, volt, topology_parameter, folder, stat_size, s))
        process.start()
    for p in procs:
        p.join()