from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation

# ─── Configuration ────────────────────────────────────────────────────────────────
N_VOLTAGES   = 50000
TIME_STEP    = 1e-11
STAT_SIZE    = 1000
N_P          = 9
CPU_CNT      = 32
U0_LIST      = np.linspace(0.01,0.05,CPU_CNT,endpoint=False)
OUTPUT_DIR   = Path("/home/j/j_mens07/phd/data/2_funding_period/")
# BASE_DIR   = Path("/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/")
LOG_LEVEL    = logging.INFO
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    time_pts    = np.arange(N_VOLTAGES) * TIME_STEP
    tasks       = []

    # Common output base for voltage sweep
    out_base = OUTPUT_DIR / "potential/wo_magic_cable/dc_input_vs_volt"
    out_base.mkdir(parents=True, exist_ok=True)

    # --- 8‐electrode setup ---
    topo = {"Nx": N_P,"Ny": N_P,"Nz": 1,
                "e_pos": [[(N_P-1)//2, 0],[0, 0],[N_P-1, 0],
                        [0, (N_P-1)//2],[N_P-1, (N_P-1)//2],
                        [0, N_P-1],[N_P-1, N_P-1],[(N_P-1)//2, N_P-1]],
                "electrode_type": ['constant']*8}
    n_elec = len(topo["e_pos"])
    for U0 in U0_LIST:
        volt        = np.zeros((N_VOLTAGES, n_elec+1), float)
        volt[:, 0]  = U0
        out_base.mkdir(exist_ok=True)
        args    = (time_pts, volt, topo, out_base)
        kwargs  = {
            'net_kwargs': {'add_to_path' : f"_{U0:.3f}"},
            'sim_kwargs': {'stat_size':STAT_SIZE,'save':True}
        }
        tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()
