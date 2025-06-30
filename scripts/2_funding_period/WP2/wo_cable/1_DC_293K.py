from pathlib import Path
import logging, numpy as np
import sys
sys.path.append("src/")
import nanonets_utils

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL      = 293.0
V_DRIVE    = 1.0
N_VOLT     = 50000
TIME_STEP  = 1e-10
STAT_SIZE  = 200
NP_LIST    = [2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]
BASE_DIR   = Path("/home/j/j_mens07/phd/data/2_funding_period/")
# BASE_DIR   = Path("/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/")
LOG_LEVEL  = logging.INFO
CPU_CNT    = 10
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    time_pts   = np.arange(N_VOLT) * TIME_STEP
    tasks      = []

    for e_types, sub in [
        (["constant","constant"], "current/wo_magic_cable/dc_input_vs_size/293"),
        (["constant","floating"], "potential/wo_magic_cable/dc_input_vs_size/293")
    ]:
        out_base = BASE_DIR / sub
        for Np in NP_LIST:
            topo       = {"Nx":Np, "Ny":1, "Nz":1,
                          "e_pos":[[0,0,0],[Np-1,0,0]],
                          "electrode_type":e_types}
            
            volt        = np.zeros((N_VOLT, len(topo["e_pos"])+1), float)
            volt[:, 0]  = V_DRIVE
            out_base.mkdir(parents=True, exist_ok=True)
            args    = (time_pts, volt, topo, out_base, STAT_SIZE, T_VAL)
            kwargs  = {
                'sim_kwargs': {'high_C_output':False},
            }
            tasks.append((args, kwargs))

    nanonets_utils.batch_launch(nanonets_utils.run_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()