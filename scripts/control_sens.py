import numpy as np
import pandas as pd
import sys
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
import model
import multiprocessing

if __name__ == '__main__':

    # Voltage Sweep
    N_voltages          = 400
    N_processes         = 10
    v_rand              = np.random.uniform(low=-0.05, high=0.05, size=((N_voltages,7)))
    v_gates             = np.random.uniform(low=-0.1, high=0.1, size=N_voltages)
    index               = [i for i in range(N_voltages)]
    rows                = [index[i::N_processes] for i in range(N_processes)]
    v_delta             = 0.005
    combinations        = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[1,0],[1,2],[1,3],[1,4],[1,5],[1,6],[2,0],[2,1],[2,3],[2,4],[2,5],[2,6],
                           [3,0],[3,1],[3,2],[3,4],[3,5],[3,6],[4,0],[4,1],[4,2],[4,3],[4,5],[4,6],[5,0],[5,1],[5,2],[5,3],[5,4],[5,6],
                           [6,0],[6,1],[6,2],[6,3],[6,4],[6,5]]

    def parallel_code(thread, rows, v_rand, v_gates, N_voltages, N_processes, v_delta, combinations):

        N                           = 7
        topology_parameter          = {}
        topology_parameter["Nx"]    = N
        topology_parameter["Ny"]    = N
        topology_parameter["Nz"]    = 1
        topology_parameter["e_pos"] = [[0,0,0], [int((N-1)/2),0,0], [N-1,0,0], [0,int((N-1)/2),0], [0,N-1,0], [N-1,int((N)/2),0], [int((N)/2),(N-1),0], [N-1,N-1,0]]

        sim_dic                 = {}
        sim_dic['error_th']     = 0.05
        sim_dic['max_jumps']    = 10000000

        target_electrode    = len(topology_parameter["e_pos"]) - 1
        folder              = "/home/jonas/phd/test_runs/controls_sens/"#"/scratch/tmp/j_mens07/data/system_size_new/"
        thread_rows         = rows[thread]
        
        v_rand_new  = v_rand[thread_rows,:]
        v_gates_new = v_gates[thread_rows]

        for combi in combinations:

            for j in range(4):

                voltages            = pd.DataFrame(np.zeros((int(N_voltages/N_processes),9)))
                voltages.iloc[:,0]  = v_rand_new[:,0]
                voltages.iloc[:,1]  = v_rand_new[:,1]
                voltages.iloc[:,2]  = v_rand_new[:,2]
                voltages.iloc[:,3]  = v_rand_new[:,3]
                voltages.iloc[:,4]  = v_rand_new[:,4]
                voltages.iloc[:,5]  = v_rand_new[:,5]
                voltages.iloc[:,6]  = v_rand_new[:,6]
                voltages.iloc[:,-1] = v_gates_new
                
                if j == 1:
                    voltages.iloc[:,combi[0]] += v_delta
                elif j == 2:
                    voltages.iloc[:,combi[1]] += v_delta
                elif j == 3:
                    voltages.iloc[:,combi[0]] += v_delta
                    voltages.iloc[:,combi[1]] += v_delta
                else:
                    pass

                sim_class = model.simulation(voltages.values)
                sim_class.init_cubic(folder=folder, topology_parameter=topology_parameter)
                sim_class.run_const_voltages(target_electrode=target_electrode, sim_dic=sim_dic, save_th=1, T_val=0.0)

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,rows,v_rand,v_gates, N_voltages, N_processes, v_delta, combinations))
        process.start()