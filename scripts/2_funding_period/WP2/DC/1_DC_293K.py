from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation
from nanonets import Simulation

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
N_REF_NET  = 4
E_POS_REF  = 0
# ────────────────────────────────────────────────────────────────────────────────

# Scale Voltage ranges
def get_transfer_coeff(n):
    topo = {"Nx": n, "Ny": 1,
            "e_pos" : [[0,0], [n-1,0]],
            "electrode_type" : ['constant']*2}
    sim_class = Simulation(topology_parameter=topo)
    sim_class.build_conductance_matrix()
    sim_class.init_transfer_coeffs()
    return sim_class.get_transfer_coeffs()

def get_scaling_factor(n=4, e_pos=0):
    transf_coeff    = np.array([get_transfer_coeff(nn) for nn in NP_LIST])
    factor          = np.ones_like(transf_coeff, dtype=float)
    np.divide(transf_coeff[n,e_pos], transf_coeff, out=factor, where=transf_coeff!=0)
    return factor

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    time_pts    = np.arange(N_VOLT) * TIME_STEP
    scale       = get_scaling_factor(N_REF_NET, E_POS_REF)
    tasks       = []

    for e_types, sub in [
        (["constant","constant"], "current/wo_magic_cable/dc_input_vs_size/293"),
        (["constant","floating"], "potential/wo_magic_cable/dc_input_vs_size/293")
    ]:
        out_base = BASE_DIR / sub
        for i, Np in enumerate(NP_LIST):
            topo       = {"Nx":Np, "Ny":1,
                          "e_pos":[[0,0],[Np-1,0]],
                          "electrode_type":e_types}
            
            volt        = np.zeros((N_VOLT, len(topo["e_pos"])+1), float)
            volt[:, 0]  = V_DRIVE
            volt        *= np.hstack((scale[i,:],0.0))
            out_base.mkdir(parents=True, exist_ok=True)
            args    = (time_pts, volt, topo, out_base)
            kwargs  = {
                'net_kwargs': {},
                'sim_kwargs': {'T_val':T_VAL,'stat_size':STAT_SIZE,'save':True}
            }
            tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()