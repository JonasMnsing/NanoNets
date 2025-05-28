from pathlib import Path
import logging, numpy as np
import sys
sys.path.append("src/")
import nanonets_utils

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL           = 5.0
STAT_SIZE       = 200
N_PERIODS       = 100
N_NP            = 9
N_SAMPLES       = 1000
F0              = 1.0
F1              = 3.0
AMPLITUDE       = 0.1
U_BOUNDS        = 0.05
OUTPUT_DIR      = Path("/home/j/j_mens07/phd/data/2_funding_period/")
# OUTPUT_DIR      = Path("/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/")
LOG_LEVEL       = logging.INFO
CPU_CNT         = 10
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    
    # Setup base directory once
    out_base = OUTPUT_DIR / "potential" / "wo_magic_cable" / "ac_two_tone_signal"
    out_base.mkdir(parents=True, exist_ok=True)

    topo    = {"Nx": N_NP,"Ny": N_NP,"Nz": 1,
               "e_pos": [[(N_NP-1)//2, 0, 0],[0, 0, 0],[N_NP-1, 0, 0],
                         [0, (N_NP-1)//2, 0],[N_NP-1, (N_NP-1)//2, 0],
                         [0, N_NP-1, 0],[N_NP-1, N_NP-1, 0],[(N_NP-1)//2, N_NP-1, 0]],
                "electrode_type": ['constant']*7 + ['floating']}

    # Control Voltage Sample
    control_sample = np.random.uniform(-U_BOUNDS,U_BOUNDS,(N_SAMPLES,5))
    
    # Generate base waveform
    f0_hz   = F0 * 1e6
    f1_hz   = F1 * 1e6
    dt      = 1 / (40 * f1_hz)
    T_sim   = N_PERIODS / f1_hz
    N_steps = int(np.ceil(T_sim / dt))

    time_steps, base_volt = nanonets_utils.sinusoidal_voltages(
        N_steps, topo,
        amplitudes=[AMPLITUDE,0.,AMPLITUDE,0.,0.,0.,0.,0.],
        frequencies=[f0_hz,0.,f1_hz,0.,0.,0.,0.,0.],
        time_step=dt
    )
    
    # Build tasks
    tasks   = []
    for i, controls in enumerate(control_sample):

        # make a fresh copy of the base waveform
        volt_i = base_volt.copy()
        # apply 5‐element control vector to the correct electrodes
        volt_i[:, [1,3,4,5,6]] = controls

        args    = (time_steps, volt_i, topo, out_base, STAT_SIZE, T_VAL)
        kwargs  = {
            'sim_kwargs': {'high_C_output'  :   False,
                            'add_to_path'   :   f"_{i}"}
        }
        tasks.append((args, kwargs))

    nanonets_utils.batch_launch(nanonets_utils.run_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()