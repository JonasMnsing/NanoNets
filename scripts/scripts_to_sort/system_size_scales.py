import numpy as np
import pandas as pd
import sys
# sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
import nanonets
import multiprocessing

if __name__ == '__main__':

    # Voltage Sweep
    N_voltages          = 80000#80640
    N_processes         = 10#36

    index               = [i for i in range(N_voltages)]
    rows                = [index[i::N_processes] for i in range(N_processes)]

    scales              = [0.69, 0.77, 0.84, 0.92, 1., 1.08, 1.16, 1.23, 1.31, 1.39, 1.47, 1.55, 1.62, 1.7, 1.78, 1.86]
    v_rand              = np.repeat(np.random.uniform(low=-0.1, high=0.1, size=((int(N_voltages/4),5))), 4, axis=0)
    v_gates             = np.repeat(np.random.uniform(low=-0.1, high=0.1, size=int(N_voltages/4)),4)
    i1                  = np.tile([0.0,0.0,0.05,0.05], int(N_voltages/4))
    i2                  = np.tile([0.0,0.05,0.0,0.05], int(N_voltages/4))

    def parallel_code(thread, rows, v_rand, v_gates, i1, i2, scales, N_voltages):

        for i, N in enumerate(range(3,11)):
            
            topology_parameter          = {}
            topology_parameter["Nx"]    = N
            topology_parameter["Ny"]    = N
            topology_parameter["Nz"]    = 1
            topology_parameter["e_pos"] = [[0,0,0], [int((N-1)/2),0,0], [N-1,0,0],
                                           [0,int((N-1)/2),0], [0,N-1,0], [N-1,int((N)/2),0],
                                           [int((N)/2),(N-1),0], [N-1,N-1,0]]

            sim_dic                 = {}
            sim_dic['error_th']     = 0.05
            sim_dic['max_jumps']    = 10000000

            target_electrode    = len(topology_parameter["e_pos"]) - 1
            # folder              = "/scratch/tmp/j_mens07/data/system_size_new/"
            folder              = "/mnt/c/Users/jonas/Desktop/phd/data/system_size_scale/"
            thread_rows         = rows[thread]
            voltages            = pd.DataFrame(np.zeros((N_voltages,9)))
            voltages.iloc[:,0]  = scales[i] * v_rand[:,0]
            voltages.iloc[:,1]  = scales[i] * i1
            voltages.iloc[:,2]  = scales[i] * v_rand[:,1]
            voltages.iloc[:,3]  = scales[i] * i2
            voltages.iloc[:,4]  = scales[i] * v_rand[:,2]
            voltages.iloc[:,5]  = scales[i] * v_rand[:,3]
            voltages.iloc[:,6]  = scales[i] * v_rand[:,4]
            voltages.iloc[:,-1] = v_gates

            sim_class = nanonets.simulation(voltages.values[thread_rows,:])
            sim_class.init_cubic(folder=folder, topology_parameter=topology_parameter)
            sim_class.run_const_voltages(target_electrode=target_electrode, sim_dic=sim_dic, save_th=20, T_val=0.0)

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,rows,v_rand,v_gates, i1, i2, scales, N_voltages))
        process.start()