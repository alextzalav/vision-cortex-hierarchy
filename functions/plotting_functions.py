import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from matplotlib.lines import Line2D

groupDirAmarel = '/projects/f_mc1689_1/'
actflow_toolbox_path = groupDirAmarel + 'AnalysisTools/'
sys.path.insert(0, actflow_toolbox_path)
import ActflowToolbox as actflow

networkpartition_dir = '/projects/f_mc1689_1/AnalysisTools/ActflowToolbox/dependencies/ColeAnticevicNetPartition/'
networkdef = np.loadtxt(networkpartition_dir + 'cortex_parcel_network_assignments.txt') # cortical parcellation; networkdef is a 718-element vector with values 1-11 denoting       each region's network
networkorder = np.asarray(sorted(range(len(networkdef)), key=lambda k: networkdef[k])) # derive networkorder i.e. specific indices to reorder matrix into network communities
orderedNetworks = ['VIS1','VIS2','SMN','CON','DAN','LAN','FPN','AUD','DMN','PMM','VMM','OA'] # order of networks i.e. VIS1 network first -> OA network last
networkpalette = ['royalblue','slateblue','paleturquoise','darkorchid','limegreen',
                  'lightseagreen','yellow','orchid','r','peru','orange','olivedrab'] # canonical colors for each network
networkpalette = np.asarray(networkpalette)


def plot_glasso(glasso_matrix, glasso_folder):    
    
    fcmat_cort_ordered=glasso_matrix[networkorder,:][:,networkorder]
    plt.figure(figsize = (10,5))
    fig=actflow.tools.addNetColors_Seaborn(fcmat_cort_ordered)

    if glasso_folder:
        os.chdir(glasso_folder)
        fig.savefig("avg_glasso_15_subs.svg", format='svg')
        fig.savefig("avg_glasso_15_subs.png", format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)

    plt.close(fig)


def draw_hemisphere_boxplot(
    metric, metric_name, names, title,
    save_path=None, figsize=(12,12),
    base_fs=12, sig_pairs=None, showmeans=True
):
    """
    Draws a polished hemisphere boxplot using Seaborn’s boxplot,
    with manual gray dots for subjects and significance brackets.
    """
    # If only one region, make names a tuple to avoid grouping issues
    if len(names) == 1:
        names = (names[0],)

    # Plain white style
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize)
    sns.despine(ax=ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Color mapping for boxes
    set2 = sns.color_palette("Set2", n_colors=len(names))
    if metric.shape[1] == 4:
        cmap = {names[0]: set2[3], names[1]: set2[0],
                names[2]: set2[1], names[3]: set2[2]}
    elif metric.shape[1] == 3:
        cmap = {names[0]: set2[0], names[1]: set2[1], names[2]: set2[2]}
    else:
        cmap = {}
    palette = [cmap.get(r, "lightblue") for r in names]

    # Boxplot
    sns.boxplot(data=metric, palette=palette, ax=ax,
                showfliers=False,
                medianprops={'color':'orange','linewidth':2})
    ax.grid(False)

    # Manual scatter for subject points (consistent gray)
    n_regions = metric.shape[1]
    for i in range(n_regions):
        vals = metric[:, i]
        # jitter around x = i
        xs = np.random.normal(loc=i, scale=0.05, size=len(vals))
        ax.scatter(xs, vals, color='gray', alpha=0.7, s=20, zorder=3)

    # Legend for mean & median
    xs_centers = np.arange(metric.shape[1])
    handles = []
    if showmeans:
        means = np.nanmean(metric, axis=0)
        ax.scatter(xs_centers, means, marker="^", color="green",
                   edgecolors="black", s=50, label="Mean", zorder=4)
        handles.append(Line2D([], [], marker="^", color="green",
                              markeredgecolor="black", linestyle="",
                              label="Mean"))
    # median line in legend
    handles.append(Line2D([], [], color="orange", linewidth=2, label="Median"))
    ax.legend(handles=handles, frameon=False,
              loc="upper right", bbox_to_anchor=(1.2,1),
              fontsize=base_fs-4)

    # Significance brackets
    if sig_pairs:
        y_min, y_max = np.nanmin(metric), np.nanmax(metric)
        yrange = y_max - y_min
        offset, h = 0.09 * yrange, 0.01 * yrange

        for idx, (r1, r2, p_val) in enumerate(sig_pairs):
            if p_val < 0.05:
                x1, x2 = names.index(r1), names.index(r2)
                stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
                y = y_max + offset * (idx + 1)
                ax.plot([x1, x1, x2, x2],
                        [y, y+h, y+h, y],
                        lw=1.5, color="black")
                ax.text((x1 + x2) / 2, y + h, stars,
                        ha="center", va="bottom",
                        fontsize=base_fs-4)
        ax.set_ylim(top=y_max + offset * (len(sig_pairs) + 2))

    # Final labels
    ax.set_xticklabels(names, fontsize=base_fs, rotation=0)
    ax.set_title(title, fontsize=base_fs+2)
    ax.set_ylabel(metric_name, fontsize=base_fs)
    ax.tick_params(axis='y', labelsize=base_fs)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    



    
def plot_heatmap(
    matrix,         # ndarray, shape (n_regions, n_regions), values in 0–100
    names,            # list of length n_regions
    title,            # string for the plot title
    colourbar_name,
    vmin=0,
    vmax=100,
    save_path=None,   # filepath to save (e.g. 'figs/connectivity.pdf')
    figsize=(7,7),    # figure size in inches
    base_fs=12,       # base font size
    cmap='coolwarm'    # colormap (try 'viridis', 'Blues', 'magma', etc.)
):
    """
    matrix :            symmetric matrix of connection percentages between parcels
    names    :          parcel names (tick labels)
    title    :          figure title
    colourbar_name:     name of the bar depending on what we plot
    save_path:          if provided, saves as PDF (vector)
    cmap:               best option is coolwarm or magma 
    """

    fig, ax = plt.subplots(figsize=figsize)

    # 2) Despine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 3) Show matrix
    im = ax.imshow(
        matrix,
        interpolation='none',
        cmap=cmap,
        vmin=vmin, vmax=vmax,    # since these are percentages
        aspect='equal'
    )

    # 4) Ticks & labels
    n = len(names)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=base_fs)
    ax.set_yticklabels(names, fontsize=base_fs)

    # 5) Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(colourbar_name, fontsize=base_fs)
    cbar.ax.tick_params(labelsize=base_fs)

    # 6) Title
    ax.set_title(title, fontsize=base_fs+2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
