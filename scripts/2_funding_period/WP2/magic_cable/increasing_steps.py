import sys
sys.path.append("src/")
import nanonets
import numpy as np
import multiprocessing

def run_simulation(time_steps, voltages, topology_parameter, folder, stat_size, cap):

    np_info2    = {
        'np_index'      : [10], 
        'mean_radius'   : cap,
        'std_radius'    : 0.0
    }
    
    target_electrode    = len(topology_parameter["e_pos"])-1
    sim_class           = nanonets.simulation(topology_parameter=topology_parameter, folder=folder, high_C_output=True, np_info2=np_info2, add_to_path=f'_{cap}')
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, stat_size=stat_size, save=True, T_val=5.0)

if __name__ == '__main__':

    folder              = "/home/j/j_mens07/phd/data/2_funding_period/potential/magic_cable/increasing_steps/"
    topology_parameter  = {
        "Nx"                : 10,
        "Ny"                : 1,
        "Nz"                : 1,
        "e_pos"             : [[0,0,0],[9,0,0]],
        "electrode_type"    : ['constant','floating']
    }
    
    # Voltages
    U_0             = 0.02
    steps_per_step  = 40000
    U_i1            = U_0*np.repeat(np.arange(6),steps_per_step)
    U_i2            = U_0*np.repeat(np.arange(4,-1,-1),steps_per_step)
    U_i             = np.hstack((U_i1,U_i2))
    N_voltages      = len(U_i)
    volt            = np.zeros(shape=(len(U_i),3))
    volt[:,0]       = U_i

    time_step   = 1e-10
    stat_size   = 50
    cap         = 5e3
    time_steps  = np.arange(N_voltages)*time_step
    
    run_simulation(time_steps, volt, topology_parameter, folder, stat_size, cap)

    # N_processes = len(cap_vals)
    # procs       = []
    
    # for i in range(N_processes):
    #     cap     = cap_vals[i]
    #     process = multiprocessing.Process(target=run_simulation, args=(time_steps, volt, topology_parameter, folder, stat_size, cap))
    #     process.start()
    # for p in procs:
    #         p.join()

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
    #     procs.append(process)
    # for p in procs:
    #         p.join()