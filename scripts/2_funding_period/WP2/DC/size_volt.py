from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation

# ─── Configuration ────────────────────────────────────────────────────────────────
N_VOLTAGES   = 50000
TIME_STEP    = 1e-11
STAT_SIZE    = 500
N_P_VALUES   = [5,7,11,13]
CPU_CNT      = 32
U0_LIST      = np.linspace(0.01,0.05,CPU_CNT,endpoint=False)
OUTPUT_DIR   = Path("/scratch/j_mens07/data/2_funding_period/dynamic/DC/size_volt/")
LOG_LEVEL    = logging.INFO
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    time_pts    = np.arange(N_VOLTAGES) * TIME_STEP
    tasks       = []

    # Common output base for voltage sweep
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for N_P in N_P_VALUES:
        # --- 8‐electrode (8 constant) setup ---
        topo = {"Nx": N_P,"Ny": N_P,
                    "e_pos": [[(N_P-1)//2, 0],[0, 0],[N_P-1, 0],
                            [0, (N_P-1)//2],[N_P-1, (N_P-1)//2],
                            [0, N_P-1],[N_P-1, N_P-1],[(N_P-1)//2, N_P-1]],
                    "electrode_type": ['constant']*8}
        n_elec = len(topo["e_pos"])
        for U0 in U0_LIST:
            volt        = np.zeros((N_VOLTAGES, n_elec+1), float)
            volt[:, 0]  = U0
            args    = (time_pts, volt, topo, OUTPUT_DIR)
            kwargs  = {
                'net_kwargs': {'add_to_path' : f"_{U0:.5f}"},
                'sim_kwargs': {'stat_size':STAT_SIZE,'save':True}
            }
            tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()
