import logging
import multiprocessing
from pathlib import Path

import numpy as np
import nanonets_utils
from core import run_simulation, batch_launch

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL           = 5.0
STAT_SIZE_BASE  = 10
N_PERIODS       = 100
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

    # Define two circuits: closed (constant both electrodes) and open (last floating)
    circuits = [
        (['constant', 'constant'], OUTPUT_DIR / "current" / "wo_magic_cable" / "ac_input_vs_freq"),
        (['constant', 'floating'], OUTPUT_DIR / "potential" / "wo_magic_cable" / "ac_input_vs_freq")
    ]

    for electrode_types, out_base in circuits:
        out_base.mkdir(parents=True, exist_ok=True)

        # shared topology for this circuit
        topo = {
            "Nx": N_NP,
            "Ny": 1,
            "Nz": 1,
            "e_pos": [[0, 0, 0], [N_NP-1, 0, 0]],
            "electrode_type": electrode_types
        }

        for freq_mhz in FREQ_LIST_MHZ:
            f0_hz       = freq_mhz * 1e6
            dt          = 1/(40 * f0_hz) 
            T_sim       = N_PERIODS / f0_hz
            N_steps     = int(np.ceil(T_sim / dt))
            stat_size   = max(STAT_SIZE_BASE, int(np.round(300*freq_mhz/5.0)))

            # generate sinusoidal waveforms for all electrodes
            time_steps, volt = nanonets_utils.sinusoidal_voltages(
                N_steps,
                topo,
                amplitudes=[AMPLITUDE, 0.0],
                frequencies=[f0_hz, 0.0],
                time_step=dt
            )

            # output directory per frequency
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

    logging.info(f"Launching {len(tasks)} AC-frequency simulations on up to {cpu_cnt} cores…")
    batch_launch(run_simulation, tasks, max_procs=cpu_cnt)
    logging.info("All AC-frequency simulations complete.")


if __name__ == "__main__":
    main()
