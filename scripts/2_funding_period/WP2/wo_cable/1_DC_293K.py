from pathlib import Path
import logging, multiprocessing, numpy as np
from core import run_simulation, batch_launch, build_voltage_matrix

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL      = 293.0
V_DRIVE    = 0.1
N_VOLT     = 50000
TIME_STEP  = 1e-10
STAT_SIZE  = 200
NP_LIST    = [2,4,6,8,10,12,14,16,18,20]
# NP_LIST    = [22,24,26,28,30,32,34,36,38,40]
BASE_DIR   = Path("/home/j/j_mens07/phd/data/2_funding_period/")
# BASE_DIR   = Path("/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/")
LOG_LEVEL  = logging.INFO
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    cpu_cnt    = multiprocessing.cpu_count()
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
            volt       = build_voltage_matrix(N_VOLT, len(topo["e_pos"]), V_DRIVE)
            out_base.mkdir(parents=True, exist_ok=True)
            tasks.append((time_pts, volt, topo, out_base, STAT_SIZE, T_VAL))

    batch_launch(run_simulation, tasks, cpu_cnt)

if __name__ == "__main__":
    main()