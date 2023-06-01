import numpy as np
import pandas as pd
import sys
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
import simulation
import multiprocessing

if __name__ == '__main__':

    # Voltage Sweep
    N_voltages          = 4#80028
    N_processes         = 4
    v_rand              = np.repeat(np.random.uniform(low=-0.05, high=0.05, size=((int(N_voltages/4),5))), 4, axis=0)
    v_gates             = np.repeat(np.random.uniform(low=-0.1, high=0.1, size=int(N_voltages/4)),4)
    i1                  = np.tile([0.0,0.0,0.01,0.01], int(N_voltages/4))
    i2                  = np.tile([0.0,0.01,0.0,0.01], int(N_voltages/4))

    def parallel_code(thread, v_rand, v_gates, i1, i2, N_voltages):

        for N in [10,20,30,40,50,60,70,80,90,100]:
            
            topology_parameter          = {}
            topology_parameter["Np"]    = N
            topology_parameter["Nj"]    = 4
            topology_parameter["e_pos"] = [[-1.5,-1.5],[0,-1.5],[1.5,-1.5],[-1.5,0],[-1.5,1.5],[1.5,0],[0,1.5],[1.5,1.5]]

            sim_dic                 = {}
            sim_dic['error_th']     = 0.05
            sim_dic['max_jumps']    = 100#10000000

            target_electrode    = len(topology_parameter["e_pos"]) - 1
            # folder              = "/mnt/c/Users/jonas/Desktop/phd/NanoNets/test_runs/"
            folder              = "/home/jonas/phd/NanoNets/test_runs/"
            voltages            = pd.DataFrame(np.zeros((N_voltages,9)))
            voltages.iloc[:,0]  = v_rand[:,0]
            voltages.iloc[:,2]  = v_rand[:,1]
            voltages.iloc[:,4]  = v_rand[:,2]
            voltages.iloc[:,5]  = v_rand[:,3]
            voltages.iloc[:,6]  = v_rand[:,4]
            voltages.iloc[:,1]  = i1
            voltages.iloc[:,3]  = i2
            voltages.iloc[:,-1] = v_gates

            sim_class = simulation.simulation(voltages=voltages.values)
            sim_class.init_random(folder=folder, topology_parameter=topology_parameter, add_to_path=f"_{thread}")
            sim_class.run_const_voltages(target_electrode=target_electrode, save_th=1, sim_dic=sim_dic)   

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i, v_rand, v_gates, i1, i2, N_voltages))
        process.start()