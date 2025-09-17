from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL      = 5.0
VOLT_LIST  = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
N_VOLT     = 50000
TIME_STEP  = 1e-10
STAT_SIZE  = 200
NP         = 10
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
        ("closed",   "current/wo_magic_cable/dc_input_vs_volt"),
        ("open",     "potential/wo_magic_cable/dc_input_vs_volt")
    ]:
        out_base = BASE_DIR / subpath
        for V_drive in VOLT_LIST:
            topo = {"Nx": NP,"Ny": 1,
                    "e_pos": [[0, 0], [NP - 1, 0]],
                    "electrode_type": ["constant", "constant"]
                    if circuit == "closed"
                    else ["constant", "floating"]}
            volt        = np.zeros((N_VOLT, len(topo["e_pos"])+1), float)
            volt[:, 0]  = V_drive
            out_base.mkdir(parents=True, exist_ok=True)
            args    = (time_pts, volt, topo, out_base)
            kwargs  = {
                'net_kwargs': {'add_to_path' : f"_{V_drive:.3f}"},
                'sim_kwargs': {'T_val':T_VAL,'stat_size':STAT_SIZE,'save':True}
            }
            tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()