import sys
sys.path.append("src/")
import nanonets
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size, steps_between):

    np_info2    = {
        'np_index'      : [81], 
        'mean_radius'   : 5e3,
        'std_radius'    : 0.0
    }
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, high_C_output=True, np_info2=np_info2, add_to_path=f'_{steps_between}')
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    N_voltages          = 200000
    time_step           = 1e-9
    stat_size           = 50
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
                            'constant','constant','constant','floating']
    }
    folder                  = "/home/j/j_mens07/phd/data/2_funding_period/potential/magic_cable/dc_two_step_input/"
    U_0                     = 0.1
    steps_per_step          = 40000
    steps_between_storage   = [0,400,800,1600,2000,4000,5000,10000,20000,40000]
    N_processes             = len(steps_between_storage)
    procs                   = []
    
    for i in range(N_processes):
        steps_between           = steps_between_storage[i]
        volt                    = np.zeros(shape=(N_voltages,9))
        volt[:steps_per_step,0] = U_0
        volt[steps_per_step+steps_between:2*steps_per_step+steps_between,0] = U_0

        process = multiprocessing.Process(target=run_simulation, args=(time_steps, volt, topology_parameter, folder, stat_size, steps_between))
        process.start()
    for p in procs:
        p.join()