from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation, sinusoidal_voltages

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL           = 5.0
STAT_SIZE_BASE  = 10
N_PERIODS       = 100
N_NP            = 9
FREQ_LIST_MHZ   = [133.,105.,86.,68.,55.,44.,36.,28.,23.,18.,15.,12.,10.,8.,6.,5.,2.,1.,
                   0.5,0.25,0.12,0.06,0.03,0.01,0.008,0.006,0.004,0.002,0.001]
AMPLITUDE       = 0.1
OUTPUT_DIR      = Path("/home/j/j_mens07/phd/data/2_funding_period/")
# OUTPUT_DIR      = Path("/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/")
LOG_LEVEL       = logging.INFO
CPU_CNT         = 10
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    tasks   = []

    # Define two circuits: closed (constant both electrodes) and open (last floating)
    circuits = [
        (['constant', 'constant'], OUTPUT_DIR / "current" / "wo_magic_cable" / "ac_input_vs_freq"),
        (['constant', 'floating'], OUTPUT_DIR / "potential" / "wo_magic_cable" / "ac_input_vs_freq")
    ]

    for electrode_types, out_base in circuits:
        out_base.mkdir(parents=True, exist_ok=True)

        # shared topology for this circuit
        topo = {"Nx": N_NP,"Ny": 1,"Nz": 1,
                "e_pos": [[0, 0], [N_NP-1, 0]],
                "electrode_type": electrode_types}

        for freq_mhz in FREQ_LIST_MHZ:
            f0_hz       = freq_mhz * 1e6
            dt          = 1/(40 * f0_hz) 
            T_sim       = N_PERIODS / f0_hz
            N_steps     = int(np.ceil(T_sim / dt))
            stat_size   = max(STAT_SIZE_BASE, int(np.round(300*freq_mhz/5.0)))

            # generate sinusoidal waveforms for all electrodes
            time_steps, volt = sinusoidal_voltages(
                N_steps,topo, amplitudes=[AMPLITUDE, 0.0],
                frequencies=[f0_hz, 0.0],time_step=dt)

            # output directory per frequency
            out_base.mkdir(exist_ok=True)
            args    = (time_steps, volt, topo, out_base)
            kwargs  = {
                'net_kwargs': {'add_to_path' : f"_{freq_mhz:.3f}"},
                'sim_kwargs': {'T_val':T_VAL, 'stat_size':stat_size, 'save':True}
            }
            tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()
