import logging
import multiprocessing
from pathlib import Path

import numpy as np
from core import run_simulation, batch_launch, build_voltage_matrix

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL      = 5.0
VOLT_LIST  = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
N_VOLT     = 50000
TIME_STEP  = 1e-10
STAT_SIZE  = 200
NP         = 9
BASE_DIR   = Path("/home/j/j_mens07/phd/data/2_funding_period/")
# BASE_DIR   = Path("/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/")
LOG_LEVEL  = logging.INFO
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    cpu_cnt  = multiprocessing.cpu_count()
    time_pts = np.arange(N_VOLT) * TIME_STEP

    tasks = []
    for circuit, subpath in [
        ("closed",   "current/wo_magic_cable/dc_input_vs_volt"),
        ("open",     "potential/wo_magic_cable/dc_input_vs_volt")
    ]:
        out_base = BASE_DIR / subpath

        for V_drive in VOLT_LIST:
            topo = {
                "Nx": NP,
                "Ny": 1,
                "Nz": 1,
                "e_pos": [[0, 0, 0], [NP - 1, 0, 0]],
                "electrode_type": ["constant", "constant"]
                                 if circuit == "closed"
                                 else ["constant", "floating"]
            }

            volt = build_voltage_matrix(N_VOLT, len(topo["e_pos"]), V_drive)
            out_folder = out_base / f"V{V_drive:.3f}"
            out_folder.mkdir(parents=True, exist_ok=True)

            tasks.append((
                time_pts,
                volt,
                topo,
                out_folder,
                STAT_SIZE,
                T_VAL
            ))

    logging.info(f"Launching {len(tasks)} simulations on up to {cpu_cnt} cores…")
    batch_launch(run_simulation, tasks, max_procs=cpu_cnt)
    logging.info("Sweep complete.")

if __name__ == "__main__":
    main()