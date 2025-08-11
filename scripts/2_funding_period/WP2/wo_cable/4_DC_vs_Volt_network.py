from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL        = 5.0
N_VOLTAGES   = 50000
TIME_STEP    = 1e-10
STAT_SIZE    = 200
N_P          = 9
U0_LIST      = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
OUTPUT_DIR   = Path("/home/j/j_mens07/phd/data/2_funding_period/")
# BASE_DIR   = Path("/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/")
LOG_LEVEL    = logging.INFO
CPU_CNT      = 10
# ────────────────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    time_pts    = np.arange(N_VOLTAGES) * TIME_STEP
    tasks       = []

    # Common output base for voltage sweep
    out_base = OUTPUT_DIR / "potential/wo_magic_cable/dc_input_vs_volt"
    out_base.mkdir(parents=True, exist_ok=True)

    # --- 2‐electrode (one constant, one floating) setup ---
    topo_2 = {"Nx": N_P,"Ny": N_P,
              "e_pos": [[(N_P-1)//2, 0],[(N_P-1)//2, N_P-1]],
              "electrode_type": ['constant', 'floating']}
    # --- 8‐electrode (7 constant, 1 floating) setup ---
    topo_8 = {"Nx": N_P,"Ny": N_P,"Nz": 1,
                "e_pos": [[(N_P-1)//2, 0],[0, 0],[N_P-1, 0],
                        [0, (N_P-1)//2],[N_P-1, (N_P-1)//2],
                        [0, N_P-1],[N_P-1, N_P-1],[(N_P-1)//2, N_P-1]],
                "electrode_type": ['constant']*7 + ['floating']}
    for topo in [topo_2, topo_8]:
        n_elec = len(topo["e_pos"])
        for U0 in U0_LIST:
            volt        = np.zeros((N_VOLTAGES, n_elec+1), float)
            volt[:, 0]  = U0
            out_base.mkdir(exist_ok=True)
            args    = (time_pts, volt, topo, out_base)
            kwargs  = {
                'net_kwargs': {'add_to_path' : f"_{U0:.3f}"},
                'sim_kwargs': {'T_val':T_VAL,'stat_size':STAT_SIZE,'save':True}
            }
            tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()
