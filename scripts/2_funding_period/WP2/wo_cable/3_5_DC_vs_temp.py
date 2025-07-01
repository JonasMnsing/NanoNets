from pathlib import Path
import logging, numpy as np
import sys
sys.path.append("src/")
import nanonets_utils

# ─── Configuration ────────────────────────────────────────────────────────────────
V_drive    = 0.02
T_LIST     = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
N_VOLT     = 50000
TIME_STEP  = 1e-10
STAT_SIZE  = 200
NP         = 9
BASE_DIR   = Path("/home/j/j_mens07/phd/data/2_funding_period/")
# BASE_DIR   = Path("/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/")
LOG_LEVEL  = logging.INFO
CPU_CNT    = 10
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    time_pts    = np.arange(N_VOLT) * TIME_STEP
    tasks       = []

    for circuit, subpath in [
        ("closed",   "current/wo_magic_cable/dc_input_vs_temp"),
        ("open",     "potential/wo_magic_cable/dc_input_vs_temp")
    ]:
        out_base = BASE_DIR / subpath
        for T_VAL in T_LIST:
            topo = {"Nx": NP,"Ny": 1,"Nz": 1,
                    "e_pos": [[0, 0, 0], [NP - 1, 0, 0]],
                    "electrode_type": ["constant", "constant"]
                    if circuit == "closed"
                    else ["constant", "floating"]}
            volt        = np.zeros((N_VOLT, len(topo["e_pos"])+1), float)
            volt[:, 0]  = V_drive
            out_base.mkdir(parents=True, exist_ok=True)
            args    = (time_pts, volt, topo, out_base, STAT_SIZE, T_VAL)
            kwargs  = {
                'sim_kwargs': {'high_C_output'  :   False,
                               'add_to_path'    :   f"_{T_VAL:.3f}"}
            }
            tasks.append((args, kwargs))

    nanonets_utils.batch_launch(nanonets_utils.run_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()