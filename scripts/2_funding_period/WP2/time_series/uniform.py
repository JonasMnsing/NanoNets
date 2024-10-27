import multiprocessing
import numpy as np
import sys
sys.path.append("src/")

import nanonets


def run_simulation(voltages, topology, sim_dic):

    target_electrode    = len(topology["e_pos"]) - 1
    sim_class           = nanonets.simulation(network_topology='cubic', topology_parameter=topology, folder=folder+"data/", add_to_path=f'_{target_electrode}')
    sim_class.run_const_voltages(voltages=voltages, target_electrode=target_electrode, output_potential=True, verbose=True, sim_dic=sim_dic, save_th=0.1)
    np.savetxt(test.csv, sim_class.return_potential_landscape())

# Path
folder = "scripts/2_funding_period/WP2/time_series/uniform/"

# Network Parameter
N_electrodes    = 8
N_particles     = 7
topology        = {
    "Nx"    : N_particles,
    "Ny"    : N_particles,
    "Nz"    : 1,
    "e_pos" :  [[0,0,0],[int((N_particles-1)/2),0,0],[N_particles-1,0,0],[0,int((N_particles-1)/2),0],[0,N_particles-1,0],
                [int((N_particles-1)/2),N_particles-1,0],[N_particles-1,int((N_particles-1)/2),0],[N_particles-1,N_particles-1,0]],
    "electrode_type" : ['constant','floating','floating','floating','floating','floating','floating','floating']
}
sim_dic         = {
    "error_th"        : 0.05,      
    "max_jumps"       : 1000,
    "eq_steps"        : 0,
    "jumps_per_batch" : 1,
    "kmc_counting"    : False,
    "min_batches"     : 1
}

# Voltage Values
voltages                    = np.zeros(shape=(1,N_electrodes+1))
voltages[:,:N_electrodes]   = np.random.uniform(low=-0.1, high=0.1, size=(1,N_electrodes))

procs = []
for target_electrode in range(1,8):

    process = multiprocessing.Process(target=run_simulation, args=(voltages, topology, sim_dic))
    process.start()
    procs.append(process)

for p in procs:
    p.join()

