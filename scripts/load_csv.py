import numpy as np
import pandas as pd
import sys
sys.path.append("/home/jonas/phd/NanoNets/src/")
import model
import multiprocessing
import time

if __name__ == '__main__':

    N_processes         = 10
    voltages            = pd.read_csv("scripts/voltage_configs.csv", sep=' ', header=None)
    N_voltages          = len(voltages)
    index               = [i for i in range(N_voltages)]
    rows                = [index[i::N_processes] for i in range(N_processes)]
    print(voltages)
    def parallel_code(thread, rows, voltages):

        times = []

        for N in range(3,10):
            
            # Topology values
            topology_parameter          = {}
            topology_parameter["Nx"]    = N
            topology_parameter["Ny"]    = N
            topology_parameter["Nz"]    = 1
            topology_parameter["e_pos"] = [[0,0,0], [N-1,0,0], [0,N-1,0], [N-1,N-1,0]]

            # Simulation Values
            sim_dic                 = {}
            sim_dic['error_th']     = 0.005
            sim_dic['max_jumps']    = 200000000

            # Misc
            target_electrode    = len(topology_parameter["e_pos"]) - 1
            folder              = "/home/jonas/phd/NanoNets/test_runs/evan/"
            thread_rows         = rows[thread]

            t1 = time.process_time_ns()

            # Run Simulation
            sim_class = model.simulation(folder=folder, voltages=voltages.values[thread_rows,:], topology_parameter=topology_parameter)
            sim_class.run_const_voltages(target_electrode=target_electrode, T_val=0.28, sim_dic=sim_dic, save_th=10)

            t2 = time.process_time_ns()
            times.append(t2-t1)
        
        np.savetxt(fname=f"{folder}times_e_{sim_dic['error_th']}_{thread}.csv", X=np.array(times))

    for i in range(N_processes):

        process = multiprocessing.Process(target=parallel_code, args=(i,rows,voltages))
        process.start()