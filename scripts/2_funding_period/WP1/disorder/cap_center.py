import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets import Simulation
from nanonets.utils import logic_gate_sample, distribute_array_across_processes ,batch_launch, run_static_simulation

# ─── Configuration ───
N_P         = 9
N_E         = 8
V_CONTROL   = 0.1
V_INPUT     = 0.01
V_GATE      = 0.0
N_DATA      = 32000
N_PROCS     = 10
T_VAL       = 5
LOG_LEVEL   = logging.INFO
INPUT_POS   = [1,3]
E_POS_REF   = 1
# PATH        = Path("/mnt/c/Users/jonas/Desktop/phd/data/1_funding_period/system_size_scaled/")
PATH        = Path("/scratch/j_mens07/data/2_funding_period/static/cap_disorder/")
RAD_VALS    = np.load("scripts/2_funding_period/WP2/wo_cable/data/radius_dis_center.npy")
TOPO_VALS   = np.load("scripts/2_funding_period/WP2/wo_cable/data/topo_dis_center.npy")
DIST_VALS   = np.load("scripts/2_funding_period/WP2/wo_cable/data/dist_dis_center.npy")
E_DIST_VALS = np.load("scripts/2_funding_period/WP2/wo_cable/data/e_dist_dis_center.npy")
SAVE_TH     = 100
# --------------------- 

def get_transfer_coeff(n):
    topo = {"Nx": n, "Ny": n,
            "e_pos" : [[0,0], [int((n-1)/2),0], [n-1,0], [0,int((n-1)/2)],
                       [0,n-1], [n-1,int((n)/2)], [int((n)/2),(n-1)], [n-1,n-1]],
            "electrode_type" : ['constant']*8}
    sim_class = Simulation(topology_parameter=topo, pack_optimizer=False)
    sim_class.build_conductance_matrix()
    sim_class.init_transfer_coeffs()
    return sim_class.get_transfer_coeffs()

def main():
    logging.basicConfig(level=LOG_LEVEL,format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks   = []
    topo    = {"Nx": N_P, "Ny": N_P,
                "e_pos" : [[0,0], [int((N_P-1)/2),0], [N_P-1,0],
                [0,int((N_P-1)/2)], [0,N_P-1], [N_P-1,int((N_P)/2)],
                [int((N_P)/2),(N_P-1)], [N_P-1,N_P-1]],
                "electrode_type" : ['constant']*N_E}
    volt    = logic_gate_sample(V_CONTROL, INPUT_POS, N_DATA, topo, V_INPUT, V_GATE, sample_technique='uniform')
    t_coeff = get_transfer_coeff(N_P)
    scale   = np.divide(t_coeff[1], t_coeff, where=t_coeff!=0)
    volt    *= np.hstack((scale,0.0))
    kwargs  = {
        'net_kwargs': {'add_to_path' : f"_center", "net_topology" : TOPO_VALS,
                    "dist_matrix" : DIST_VALS,"electrode_dist_matrix" : E_DIST_VALS,
                    "radius_vals" : RAD_VALS,"electrode_type" : ['constant']*(E_DIST_VALS.shape[0])},
        'sim_kwargs': {'T_val':T_VAL,'save_th':SAVE_TH}
    }

    for p in range(N_PROCS):
        volt_p  = distribute_array_across_processes(p, volt, N_PROCS)
        args    = (volt_p,"",PATH)
        tasks.append((args, kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == '__main__':
    main()