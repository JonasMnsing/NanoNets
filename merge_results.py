import sys
sys.path.append("src/")
import nanonets_utils

N_stat          = 100
N_th            = 10
Nx, Ny, Nz, Ne  = 7, 7, 1, 2

for r in [1,2,3]:
    for R in [50,100,200,400,800,1600,3200,6400,12800]:

        folder  = f'/home/jonas/phd/data/1I_1O_R/R_{R}/r{r}'
        
        nanonets_utils.store_average_time_results(folder, Nx, Ny, Nz, Ne, N_stat, N_th)
        nanonets_utils.store_average_time_currents(folder, Nx, Ny, Nz, Ne, N_stat, N_th)
        nanonets_utils.store_average_time_states(folder, Nx, Ny, Nz, Ne, N_stat, N_th)