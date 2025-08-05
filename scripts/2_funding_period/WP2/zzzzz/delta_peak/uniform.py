import multiprocessing
import numpy as np
import sys
sys.path.append("src/")

import nanonets


def run_simulation(i, voltages, time_steps, topology):

    sim_class   = nanonets.simulation(topology_parameter=topology, folder=folder+"data/", add_to_path=f'_{i}')
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, eq_steps=0, stat_size=200)

# Path
folder = "scripts/2_funding_period/WP2/delta_peak/uniform/"

# Network Parameter
N_electrodes    = 8
N_particles     = 7
topology        = {
    "Nx"                : N_particles,
    "Ny"                : N_particles,
    "Nz"                : 1,
    "e_pos"             :  [[int((N_particles-1)/2),0,0],[0,0,0],[N_particles-1,0,0],[0,int((N_particles-1)/2),0],
                            [N_particles-1,int((N_particles-1)/2),0],[0,N_particles-1,0],
                            [N_particles-1,N_particles-1,0],[int((N_particles-1)/2),N_particles-1,0]],
    "electrode_type"    : ['constant','constant','constant','constant','constant','constant','constant','floating']
}

# Voltage Values
N_voltages  = 5000
max_pos     = 100
max_val     = 0.1
step_size   = 1e-10
time        = step_size*np.arange(N_voltages)
max_pos     = [1,2,4,8,16,32,64,128,256,512]

procs = []
for i in range(len(max_pos)):

    voltages                = np.zeros(shape=(N_voltages,N_electrodes+1))
    voltages[:,0]           += 0.05
    pos                     = max_pos[i]
    voltages[500:500+pos,0] = max_val

    process = multiprocessing.Process(target=run_simulation, args=(i, voltages, time, topology))
    process.start()
    procs.append(process)

for p in procs:
    p.join()

