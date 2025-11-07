from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation

# ─── Configuration ────────────────────────────────────────────────────────────────
TIME_STEP       = 1e-11
STAT_SIZE       = 1000
OUTPUT_DIR      = Path("/mnt/c/Users/jonas/Desktop/phd/nanonets/scripts/2_funding_period/WP2/AC/set/data/")
LOG_LEVEL       = logging.INFO
CPU_CNT         = 10
V_VALUES        = [0.012,0.013,0.014,0.015,0.016,0.017,0.018,0.019,0.02,0.021]
N_VOLT          = 50000
topo            = {"Nx" : 1, "Ny" : 1, "electrode_type" : ['constant','constant']}
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(exist_ok=True)
    tasks = []
    for V_DRIVE in V_VALUES:

        time_pts    = np.arange(N_VOLT) * TIME_STEP
        volt        = np.zeros((N_VOLT, len(topo["electrode_type"])+1), float)
        volt[:, 0]  = V_DRIVE
        # output directory per frequency
        args    = (time_pts, volt, topo, OUTPUT_DIR)
        kwargs  = {
            'net_kwargs': {'add_to_path' : f"_{V_DRIVE:.3f}"},
            'sim_kwargs': {'stat_size':STAT_SIZE, 'save':True}
        }
        tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()
