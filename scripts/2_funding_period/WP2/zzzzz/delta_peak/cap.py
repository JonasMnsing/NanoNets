import multiprocessing
import numpy as np
import sys
sys.path.append("src/")

import nanonets


def run_simulation(voltages, time_steps, topology, np_info, seed):
    r_Std       = np_info["std_radius"]
    sim_class   = nanonets.simulation(network_topology='cubic', topology_parameter=topology, folder=folder+"data/",add_to_path=f'_{r_Std}_{seed}', np_info=np_info, seed=seed)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=6, output_potential=True, eq_steps=0, stat_size=500)

# Path
folder = "scripts/2_funding_period/WP2/delta_peak/cap/"

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
N_voltages                          = 2000
min_val                             = 0.1
max_val                             = 0.2
delta_pos                           = 200
delta_pos2                          = 400
voltages                            = np.zeros(shape=(N_voltages,N_electrodes+1))
voltages[:,0]                       = min_val
voltages[delta_pos:delta_pos2,0]    = max_val

# Time Values
step_size   = 1e-10
time        = step_size*np.arange(N_voltages)

for r_std in [0.5,1.,1.5,2.]:

    procs = []

    for seed in range(10):

        rs      = np.random.RandomState(seed=seed)
        np_info = {
            "eps_r"         : 2.6,  # Permittivity of molecular junction 
            "eps_s"         : 3.9,  # Permittivity of oxide layer
            "mean_radius"   : 10.0, # average nanoparticle radius
            "std_radius"    : r_std,  # standard deviation of nanoparticle radius
            "np_distance"   : 1.0   # spacing between nanoparticle shells
        }

        process     = multiprocessing.Process(target=run_simulation, args=(voltages, time, topology, np_info, seed))
        process.start()
        procs.append(process)

    for p in procs:
        p.join()

