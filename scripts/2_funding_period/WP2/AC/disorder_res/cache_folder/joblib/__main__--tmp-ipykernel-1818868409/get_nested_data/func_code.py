# first line: 1
@memory.cache
def get_nested_data(r_vals, n_nets, freq_list, path, n_skip, sample_p_period, dts_dict):
    # Initialize the nested dictionaries
    y_d, y_t, y_t_e = {}, {}, {}
    slice_idx = n_skip * sample_p_period

    for R in r_vals:
        y_d[R], y_t[R], y_t_e[R] = {}, {}, {}
        
        for n in range(n_nets):
            y_d[R][n], y_t[R][n], y_t_e[R][n] = {}, {}, {}
            
            for freq in freq_list:
                # --- 1. Load Main Data File ---
                f_main = f"{path}Nx=9_Ny=9_Ne=8_{freq:.3f}_mean2_{R:.1f}_{n}.parquet"
                df_main = pd.read_parquet(f_main, columns=['Observable', 'Error'])
                
                y_t[R][n][freq] = df_main['Observable'].values[slice_idx:].copy()
                y_t_e[R][n][freq] = df_main['Error'].values[slice_idx:].copy()

                # --- 2. Load Mean State File ---
                f_mean = f"{path}mean_state_Nx=9_Ny=9_Ne=8_{freq:.3f}_mean2_{R:.1f}_{n}.parquet"
                df_mean = pd.read_parquet(f_mean)
                
                # Slicing: Rows [slice_idx:] and Columns [8:]
                mean_values = df_mean.iloc[slice_idx:, 8:].values
                y_d[R][n][freq] = get_displacement_currents(mean_values, C_US, dts_dict[freq])
                
        print(f"Done: R={R}, n={n}, freq={freq}")

    return y_d, y_t, y_t_e
