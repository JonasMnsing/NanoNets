import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets import Simulation
from nanonets.utils import logic_gate_sample, distribute_array_across_processes, batch_launch, run_static_simulation

# ─── Configuration ───
N_MIN, N_MAX    = 3, 16
N_E             = 8
V_CONTROL       = 0.1
V_INPUT         = 0.01
V_GATE          = 0.0
N_DATA          = 20000
N_PROCS         = 10
T_VAL           = 5
LOG_LEVEL       = logging.INFO
PATH            = Path("/mnt/c/Users/jonas/Desktop/phd/data/1_funding_period/system_size/unscaled/")
INPUT_POS       = [1,3]
N_REF_NET       = 9
E_POS_REF       = 1
# --------------------- 

# Scale Voltage ranges
def get_transfer_coeff(n):
    topo = {"Nx": n, "Ny": n,
            "e_pos" : [[0,0], [int((n-1)/2),0], [n-1,0], [0,int((n-1)/2)],
                       [0,n-1], [n-1,int((n)/2)], [int((n)/2),(n-1)], [n-1,n-1]],
            "electrode_type" : ['constant']*8}
    sim_class = Simulation(topology_parameter=topo)
    sim_class.build_conductance_matrix()
    sim_class.init_transfer_coeffs()
    return sim_class.get_transfer_coeffs()
def get_scaling_factor(n=9, e_pos=1):
    transf_coeff            = np.array([get_transfer_coeff(nn) for nn in range(N_MIN, N_MAX + 1)])
    factor                  = transf_coeff[n-N_MIN,e_pos]/np.array(transf_coeff)
    factor[factor==np.inf]  = 1
    return factor

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    scale = get_scaling_factor(N_REF_NET, E_POS_REF)
    tasks = []
    
    for i, n in enumerate(range(N_MIN, N_MAX + 1)):
        topo = {"Nx": n, "Ny": n,
                "e_pos" : [[0,0], [int((n-1)/2),0], [n-1,0], [0,int((n-1)/2)],
                           [0,n-1], [n-1,int((n)/2)], [int((n)/2),(n-1)], [n-1,n-1]],
                "electrode_type" : ['constant']*N_E}
        volt = logic_gate_sample(V_CONTROL, INPUT_POS, N_DATA, topo, V_INPUT,
                                 V_GATE, sample_technique='uniform')
        volt *= np.hstack((scale[i,:],0.0))
        for p in range(N_PROCS):
            volt_p  = distribute_array_across_processes(p, volt, N_PROCS)
            args    = (volt_p,topo,PATH)
            kwargs  = {}
            tasks.append((args,kwargs))

    # batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()