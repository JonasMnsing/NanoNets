import sys
sys.path.append("src/")
import nanonets
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size, steps_between):
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, add_to_path=f'_{steps_between}')
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode,
                               stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    N_voltages          = 10000
    time_step           = 1e-10
    stat_size           = 100
    time_steps          = np.arange(N_voltages)*time_step
    N_p                 = 9
    topology_parameter  = {
        "Nx"                : N_p,
        "Ny"                : N_p,
        "Nz"                : 1,
        "e_pos"             : [[(N_p-1)//2,0,0],[(N_p-1)//2,N_p-1,0]],
        "electrode_type"    : ['constant','floating']
    }
    folder                  = "/home/j/j_mens07/phd/data/2_funding_period/potential/wo_magic_cable/dc_two_step_input/"
    U_0                     = 0.1
    steps_per_step          = 1500
    # steps_between_storage   = [0,40,80,160,200,400,500,1000,2000,4000]
    # steps_between_storage   = [10,20,60,100,120,180,300,750,3000,5000]
    steps_between_storage   = [2,4,6,8,12,14,16,18,25,30]
    N_processes             = len(steps_between_storage)
    procs                   = []
    
    for i in range(N_processes):
        steps_between           = steps_between_storage[i]
        volt                    = np.zeros(shape=(N_voltages,3))
        volt[:steps_per_step,0] = U_0
        volt[steps_per_step+steps_between:2*steps_per_step+steps_between,0] = U_0

        process = multiprocessing.Process(target=run_simulation, args=(time_steps, volt, topology_parameter, folder, stat_size, steps_between))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()