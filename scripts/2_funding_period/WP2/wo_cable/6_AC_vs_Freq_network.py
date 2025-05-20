import logging
import multiprocessing
from pathlib import Path

import numpy as np
import nanonets_utils
from core import run_simulation, batch_launch

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL           = 5.0
STAT_SIZE_BASE  = 10
N_PERIODS       = 40
N_NP            = 9
# FREQ_LIST_MHZ   = [0.001,0.002,0.004,0.006,0.008,0.01,0.03,0.06,0.12]
# FREQ_LIST_MHZ   = [0.25,0.5,1.,2.,5.,6.,8.,10.,12.,15.]
FREQ_LIST_MHZ   = [18.,23.,28.,36.,44.,55.,68.,86.,105.,133.]
AMPLITUDE       = 0.1
OUTPUT_DIR      = Path("/home/j/j_mens07/phd/data/2_funding_period/")
# OUTPUT_DIR      = Path("/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/")
LOG_LEVEL       = logging.INFO
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    cpu_cnt = multiprocessing.cpu_count()
    tasks   = []

    # Define two floating-output electrode networks: 2‐electrode and 8‐electrode
    topo_list = []
    # 2‐electrode: one constant + floating
    topo_list.append({
        "Nx": N_NP,
        "Ny": 1,
        "Nz": 1,
        "e_pos": [[(N_NP-1)//2, 0, 0],[(N_NP-1)//2, N_NP-1, 0]],
        "electrode_type": ['constant', 'floating']
    })
    # 8‐electrode: 7 constant + floating
    topo_list.append({
        "Nx": N_NP,
        "Ny": 1,
        "Nz": 1,
        "e_pos": [
            [(N_NP-1)//2, 0, 0],
            [0, 0, 0],
            [N_NP-1, 0, 0],
            [0, (N_NP-1)//2, 0],
            [N_NP-1, (N_NP-1)//2, 0],
            [0, N_NP-1, 0],
            [N_NP-1, N_NP-1, 0],
            [(N_NP-1)//2, N_NP-1, 0]
        ],
        "electrode_type": ['constant']*7 + ['floating']
    })

    # single output base folder
    out_base = OUTPUT_DIR / "potential" / "wo_magic_cable" / "ac_input_vs_freq"
    out_base.mkdir(parents=True, exist_ok=True)

    # build tasks for each topology and frequency
    for topo in topo_list:
        n_elec = len(topo["e_pos"])
        for freq_mhz in FREQ_LIST_MHZ:
            f0_hz       = freq_mhz * 1e6
            dt          = 1 / (40 * f0_hz)
            T_sim       = N_PERIODS / f0_hz
            N_steps     = int(np.ceil(T_sim / dt))
            stat_size   = max(STAT_SIZE_BASE, int(np.round(300 * freq_mhz / 5.0)))

            time_steps, volt = nanonets_utils.sinusoidal_voltages(
                N_steps,
                topo,
                amplitudes=[AMPLITUDE]+(n_elec-1)*[0.0],
                frequencies=[f0_hz]+(n_elec-1)*[0.0],
                time_step=dt
            )

            out_folder = out_base / f"f{freq_mhz:.3f}MHz"
            out_folder.mkdir(exist_ok=True)

            tasks.append((
                time_steps,
                volt,
                topo,
                out_folder,
                stat_size,
                T_VAL
            ))

    logging.info(f"Launching {len(tasks)} AC-frequency simulations on up to {cpu_cnt} cores…")(f"Launching {len(tasks)} AC-frequency simulations on up to {cpu_cnt} cores…")
    batch_launch(run_simulation, tasks, max_procs=cpu_cnt)
    logging.info("All AC-frequency simulations complete.")


if __name__ == "__main__":
    main()