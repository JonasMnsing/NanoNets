import multiprocessing
import numpy as np
import sys
sys.path.append("src/")

import nanonets


def run_simulation(voltages, time_steps, target_electrode, topology):

    sim_class   = nanonets.simulation(network_topology='cubic', topology_parameter=topology, folder=folder+"data/",add_to_path=f'_{target_electrode}')
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, output_potential=True, eq_steps=0, stat_size=500)

# Path
folder = "scripts/2_funding_period/WP2/delta_peak/uniform/"

# Network Parameter
N_electrodes    = 8
N_particles     = 7
topology        = {
    "Nx"                : N_particles,
    "Ny"                : N_particles,
    "Nz"                : 1,
    "e_pos"             :  [[0,0,0],[int((N_particles-1)/2),0,0],[N_particles-1,0,0],[0,int((N_particles-1)/2),0],[0,N_particles-1,0],
                            [int((N_particles-1)/2),N_particles-1,0],[N_particles-1,int((N_particles-1)/2),0],[N_particles-1,N_particles-1,0]],
    "electrode_type"    : ['constant','floating','floating','floating','floating','floating','floating','constant']
}

# Voltage Values
N_voltages              = 2000
min_val                 = 0.1
max_val                 = 0.2
delta_pos               = 200
voltages                = np.zeros(shape=(N_voltages,N_electrodes+1)) + min_val
voltages[:,-1]          = 0
voltages[delta_pos,0]   = max_val

# Time Values
step_size   = 1e-10
time        = step_size*np.arange(N_voltages)

procs = []
for target_electrode in range(1,8):

    process = multiprocessing.Process(target=run_simulation, args=(voltages, time, target_electrode, topology))
    process.start()
    procs.append(process)

for p in procs:
    p.join()

