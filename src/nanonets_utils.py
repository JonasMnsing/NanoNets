import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import nanonets
import scienceplots

def display_network(np_network_sim : nanonets.simulation, fig=None, ax=None, blue_color='#348ABD', red_color='#A60628', save_to_path=False, 
                    style_context=['science','bright'], node_size=300, edge_width=1.0, font_size=12, title='', title_size='small',
                    arrows=False, provide_electrode_labels=None, np_numbers=False, height_scale=1, width_scale=1, margins=None):

    colors  = np.repeat(blue_color, np_network_sim.N_particles+np_network_sim.N_electrodes)
    
    if np_numbers:
        node_labels = {i : i for i in np_network_sim.G.nodes}
    else:
        node_labels = {i : '' for i in np_network_sim.G.nodes}

    if provide_electrode_labels == None:
        colors[np_network_sim.N_particles:] = red_color
    else:
        colors[np_network_sim.N_particles:] = None
    
        for i, electrode_label in enumerate(provide_electrode_labels):
            node_labels[-1-i] = electrode_label

    with plt.style.context(style_context):

        if fig == None:
            fig = plt.figure()
            fig.set_figheight(fig.get_figheight()*height_scale)
            fig.set_figwidth(fig.get_figwidth()*width_scale)
        if ax == None:
            ax = fig.add_subplot()

        ax.axis('off')
        ax.set_title(title, size=title_size)

        nx.draw_networkx(G=np_network_sim.G, pos=np_network_sim.pos, ax=ax, node_color=colors, arrows=arrows,
                         node_size=node_size, font_size=font_size, width=edge_width, labels=node_labels, clip_on=False, margins=margins)

        if save_to_path != False:

            fig.savefig(save_to_path, bbox_inches='tight', transparent=True)

    return fig, ax