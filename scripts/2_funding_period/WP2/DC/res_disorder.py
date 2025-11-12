from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation

# ─── Configuration ────────────────────────────────────────────────────────────────
N_VOLTAGES   = 50000
TIME_STEP    = 1e-11
STAT_SIZE    = 200
N_P          = 9
R_VALUES     = [50,100,200,400,800,1600]
U0           = 0.02
OUTPUT_DIR   = Path("/scratch/j_mens07/data/2_funding_period/dynamic/DC/res_disorder/")
LOG_LEVEL    = logging.INFO
CPU_CNT      = 32
N_J_TOTAL    = 2*N_P*(N_P-1)
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    time_pts    = np.arange(N_VOLTAGES) * TIME_STEP
    tasks       = []

    # Common output base for voltage sweep
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 8‐electrode (8 constant) setup ---
    topo = {"Nx": N_P,"Ny": N_P,
                "e_pos": [[(N_P-1)//2, 0],[0, 0],[N_P-1, 0],
                        [0, (N_P-1)//2],[N_P-1, (N_P-1)//2],
                        [0, N_P-1],[N_P-1, N_P-1],[(N_P-1)//2, N_P-1]],
                "electrode_type": ['constant']*8}
    n_elec = len(topo["e_pos"])

    for MEAN_R2 in R_VALUES:
        res_info2 = {'N':N_J_TOTAL//3, 'mean_R':MEAN_R2, 'std_R':0.0}
        for i in range(CPU_CNT):
            volt        = np.zeros((N_VOLTAGES, n_elec+1), float)
            volt[:, 0]  = U0
            args    = (time_pts, volt, topo, OUTPUT_DIR)
            kwargs  = {
                'net_kwargs': {'add_to_path' : f"_mean2_{MEAN_R2}_{i}", "res_info2":res_info2, "seed":i},
                'sim_kwargs': {'stat_size':STAT_SIZE, 'save':True}
            }
            tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()
