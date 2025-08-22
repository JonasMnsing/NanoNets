import logging
from pathlib import Path
import numpy as np
import pandas as pd
from nanonets.utils import batch_launch, run_dynamic_simulation, sinusoidal_voltages

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL       = 5.0
V_DRIVE     = 0.02
N_VOLT      = 50000
STAT_SIZE   = 200
TIME_STEP   = 1e-10
STAT_SIZE   = 500
OUTPUT_DIR  = Path("/home/j/j_mens07/phd/data/2_funding_period/")
LOG_LEVEL   = logging.INFO
CPU_CNT     = 10
RAD_VALS    = [np.load("scripts/2_funding_period/WP2/wo_cable/data/radius_dis.npy")[i] for i in range(CPU_CNT)]
TOPO_VALS   = [np.load("scripts/2_funding_period/WP2/wo_cable/data/topo_dis.npy")[i] for i in range(CPU_CNT)]
DIST_VALS   = [np.load("scripts/2_funding_period/WP2/wo_cable/data/dist_dis.npy")[i] for i in range(CPU_CNT)]
E_DIST_VALS = [np.load("scripts/2_funding_period/WP2/wo_cable/data/e_dist_dis.npy")[i] for i in range(CPU_CNT)]
TOPO        = {""}
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,format="%(asctime)s %(levelname)s: %(message)s")
    tasks   = []

    # single output base folder
    out_base = OUTPUT_DIR / "potential" / "wo_magic_cable" / "dc_input_disorder"
    out_base.mkdir(parents=True, exist_ok=True)

    time_pts = np.arange(N_VOLT) * TIME_STEP

    # build tasks for each topology and frequency
    for i in range(CPU_CNT):
        volt        = np.zeros((N_VOLT, E_DIST_VALS[i].shape[0]+1), float)
        volt[:, 0]  = V_DRIVE
        out_base.mkdir(parents=True, exist_ok=True)
        args = (time_pts, volt, "", out_base)

        kwargs  = {
            'net_kwargs': {'add_to_path' : f"_{i}", "net_topology" : TOPO_VALS[i],
                           "dist_matrix" : DIST_VALS[i],"electrode_dist_matrix" : E_DIST_VALS[i],
                           "radius_vals" : RAD_VALS[i],"electrode_type" : ['constant']*(E_DIST_VALS[i].shape[0]-1) + ['floating']},
            'sim_kwargs': {'T_val':T_VAL,'stat_size':STAT_SIZE,'save':True}
        }
        tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()