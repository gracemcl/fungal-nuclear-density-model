import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import joypy 
import pickle

mpl.rcParams['pdf.fonttype']   = 42 
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.sans-serif'] = ['Arial']


def solution_matrix(theta, order_all_t):
    ''' plot solution matrix '''

    fig, ax = plt.subplots(figsize=(6, 4))
    c = ax.pcolormesh(np.transpose(theta), cmap='twilight_shifted', rasterized=True)
    fig.colorbar(c, ax=ax, label='phase (0,2π)')
    ax.set_title('Solution matrix', fontsize=16)
    ax.set_xlabel('Time step', fontsize=16)
    ax.set_ylabel('Oscillator ID', fontsize=16)
    ax.set_ylim(0, len(order_all_t[-1]))
    
    return fig, ax




def plot_phases(theta, t, order_all_t):
    '''
    Plot phases through time. 
    
    This is using "order" to give the oscillators an arbitrary 
    spatial ordering for visualization (see cell_cycle_model for details on order). 
    
    '''

    N = theta.shape[1]
    to_plot = np.zeros((len(t), N))
    for i, order_i in enumerate(order_all_t):
        order_int = np.array(order_i, dtype=int)
        re_ordered = theta[i][order_int]
        to_plot[i] = np.pad(re_ordered, (0, N - len(re_ordered)), mode='constant')

    fig, ax = plt.subplots(figsize=(6, 4))
    c = ax.pcolormesh(np.transpose(to_plot), cmap='twilight_shifted', rasterized=True)
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=12)
    ax.tick_params(labelsize=12)
    ax.set_ylim(0, len(order_all_t[-1]))

    ax.set_title('Giving 1D locations to oscillators for visualization \n(daughter nuclei take place of parent)', fontsize=16)
    ax.set_xlabel('Time step', fontsize=16)
    ax.set_ylabel('Location of oscillators on line \n(not used in model)', fontsize=16)

    return fig, ax


def vary_lambda(name, panels, slope_y_lim):

    ######
    with open(f"{name}.pkl", "rb") as f:
        data = pickle.load(f)

    slopes = data["slopes"]
    stds = data["stds"]
    sync = data["sync"]
    cycles = data["cycles"]
    all_split_times = data["all_split_times"]
    #######

    def lambda_box_plot(vals):
        fig, ax = plt.subplots(figsize=(2,3))
        for i, group in enumerate(vals):
            x = np.random.normal(loc=escape_rates[i], scale=0.2*escape_rates[i], size=len(group))
            ax.scatter(x, group, s=10, facecolors=colors[i], alpha=0.2, label=escape_rates[i])
            ax.boxplot(
                group,
                positions=[escape_rates[i]],   
                widths=1*escape_rates[i],                
                notch=False,
                vert=True,
                showfliers=False,
                boxprops=dict(color=colors[i], linewidth=1.5),
                whiskerprops=dict(color=colors[i], linewidth=1.5),
                capprops=dict(color=colors[i], linewidth=1.5),
                medianprops=dict(color='black', linewidth=1)
            )
        ax.set_xscale('log')
        ax.set_xticks(escape_rates)
        ax.set_xticklabels([str(v) for v in escape_rates])
        return fig, ax

    escape_rates = [5, 0.5, 0.05, 0.005]
    colors = ['#a6bddb', '#74a9cf', '#0570b0', '#045a8d' ]

    ###################################
    ### slope:
    ###################################
    fig, ax = lambda_box_plot(slopes)

    ax.set_ylim(-0.045, slope_y_lim)
    ax.set_yticks([0, 0.05, 0.1])
    ax.set_title('slopes')

    plt.savefig(f'{name}{panels}slope.pdf', dpi=300, format='pdf', bbox_inches='tight')

    ###################################
    ### cycle lens:
    ###################################
    fig, ax = lambda_box_plot(cycles)
    plt.axhline(120, color = '#8bc34a')

    ax.set_ylim(0, 230)
    ax.set_title('cycle lens')

    plt.savefig(f'{name}{panels}lens.pdf', dpi=300, format='pdf', bbox_inches='tight')

    ###################################
    ### sync:
    ###################################
    fig, ax = lambda_box_plot(sync)

    ax.set_ylim(0, 1)
    ax.set_title('SI')

    plt.savefig(f'{name}{panels}sync.pdf', dpi=300, format='pdf', bbox_inches='tight')

    ###################################
    ### std:
    ###################################
    fig, ax = lambda_box_plot(stds)
    
    ax.set_ylim(0, 4.1)
    ax.set_yticks([0,1,2,3,4])
    ax.set_title('stds')

    plt.savefig(f'{name}{panels}std.pdf', dpi=300, format='pdf', bbox_inches='tight')


################################################
################################################

def vary_N(name, panels, slope_y_lim, just_slopes):
    ######
    with open(f"{name}.pkl", "rb") as f:
        data = pickle.load(f)

    slopes = data["slopes"]
    if just_slopes == False:
        stds = data["stds"]
        sync = data["sync"]
        cycles = data["cycles"]
        all_split_times = data["all_split_times"]

    ######

    def N_box_plot(vals):
        fig, ax = plt.subplots(figsize=(2,3))
        for i, group in enumerate(vals):
            x = np.random.normal(loc=startNs[i], scale=2, size=len(group))
            ax.scatter(x, group, s=10, facecolors=colors[i], alpha=0.2, label=startNs[i])
            ax.boxplot(
                group,
                positions=[startNs[i]],  
                widths=10,                 
                notch=False,
                vert=True,
                showfliers=False,
                boxprops=dict(color=colors[i], linewidth=1.5),
                whiskerprops=dict(color=colors[i], linewidth=1.5),
                capprops=dict(color=colors[i], linewidth=1.5),
                medianprops=dict(color='black', linewidth=1)
            )
        return fig, ax

    #####################
    # slopes:
    #####################
    colors = ["#b9bae4", '#807dba', '#54278f', "#431f72"]
    startNs = [40, 60, 80, 100]

    fig, ax = N_box_plot(slopes)

    ax.set_title('Trend in nuclear\ndensity through time', fontsize=12)
    ax.set_xlabel('Initial N', fontsize=12)
    ax.set_ylabel('Slope of nuclear\ndensity best fit line', fontsize=12)
    ax.set_xticks(startNs)
    ax.set_yticks([0, 0.05, 0.1, 0.15])
    ax.set_ylim(-0.045, slope_y_lim)
    ax.set_xlim(30, 110)

    plt.savefig(f'{name}{panels}slopes.pdf', dpi=300, format='pdf', bbox_inches='tight')

    if just_slopes == False:

        fig, ax = N_box_plot(sync)
        ax.set_title('SI', fontsize=12)
        ax.set_xlabel('initial n', fontsize=12)
        ax.set_ylabel('SI', fontsize=12)
        ax.set_xticks(startNs)
        ax.set_ylim(-0.045, 1)

        plt.savefig(f'{name}{panels}sync.pdf', dpi=300, format='pdf', bbox_inches='tight')

        fig, ax = N_box_plot(cycles)

        plt.axhline(120, color = '#8bc34a')

        ax.set_title('cycle lens', fontsize=12)
        ax.set_xlabel('initial n', fontsize=12)
        ax.set_ylabel('cycle lens', fontsize=12)
        ax.set_xticks(startNs)
        ax.set_ylim(0, 230)

        plt.savefig(f'{name}{panels}lens.pdf', dpi=300, format='pdf', bbox_inches='tight')

################################################
################################################

def vary_freqs(name, panels, slope_y_lim, means, lambda_val):
    ######
    with open(f"{name}.pkl", "rb") as f:
        data = pickle.load(f)

    slopes = data["slopes"]
    stds = data["stds"]
    sync = data["sync"]
    cycles = data["cycles"]
    # all_split_times = data["all_split_times"]

    #######

    def lambda_box_plot(vals):
        fig, ax = plt.subplots(figsize=(2,3))
        if lambda_val == 0.05:
            widths = 4
        if lambda_val == 0.005:
            widths = 12
        for i, group in enumerate(vals):
            x = np.random.normal(loc=means[i], scale=1, size=len(group))
            ax.scatter(x, group, s=10, facecolors=colors[i], alpha=0.2, label=means[i])
            ax.boxplot(
                group,
                positions=[means[i]],   
                widths=widths,                 
                notch=False,
                vert=True,
                showfliers=False,
                boxprops=dict(color=colors[i], linewidth=1.5),
                whiskerprops=dict(color=colors[i], linewidth=1.5),
                capprops=dict(color=colors[i], linewidth=1.5),
                medianprops=dict(color='black', linewidth=1)
            )
        ax.set_xticks(means)
        return fig, ax


    if lambda_val == 0.05:
        colors = ['#0570b0','#0570b0','#0570b0','#0570b0','#0570b0','#0570b0']
    if lambda_val == 0.005:
        colors = ['#045a8d', '#045a8d', '#045a8d', '#045a8d', '#045a8d', '#045a8d'] #0.005


    ###################################
    ### slope:
    ###################################
    fig, ax = lambda_box_plot(slopes)

    ax.set_ylim(-0.045, slope_y_lim)
    ax.set_yticks([0, 0.05, 0.1])
    ax.set_title('slopes')

    plt.savefig(f'{name}{panels}slope.pdf', dpi=300, format='pdf', bbox_inches='tight')

    ###################################
    ### cycle lens:
    ###################################
    fig, ax = lambda_box_plot(cycles)

    plt.axhline(120, color = '#8bc34a')

    ax.set_ylim(0, 230)
    ax.set_title('cycle lens')

    plt.savefig(f'{name}{panels}lens.pdf', dpi=300, format='pdf', bbox_inches='tight')


    ###################################
    ### sync:
    ###################################
    fig, ax = lambda_box_plot(sync)

    ax.set_ylim(0, 1)
    ax.set_title('SI')

    plt.savefig(f'{name}{panels}sync.pdf', dpi=300, format='pdf', bbox_inches='tight')


    ###################################
    ### std:
    ###################################
    fig, ax = lambda_box_plot(stds)
    
    ax.set_ylim(0, 6)
    ax.set_title('stds')

    plt.savefig(f'{name}{panels}std.pdf', dpi=300, format='pdf', bbox_inches='tight')


################################################
################################################

def ridge_plot(name, panels, vary, seeds):

    with open(f"{name}.pkl", "rb") as f:
        data = pickle.load(f)
    all_split_times = data["all_split_times"]

    # loading experimental data:
    with open(f"all_intervals.pkl", "rb") as f:
        all_intervals = pickle.load(f)

    if vary == 'lambda':
        labels = [5, 0.5, 0.05, 0.005]
        colors = ['#a6bddb', '#74a9cf', '#0570b0', '#045a8d']
    if vary == 'N':
        labels = [40, 60, 80, 100]
        colors = ["#b9bae4", '#807dba', '#54278f', "#431f72"]

    ###################################
    ### branch intervals
    ###################################

    split_times_escape = [[] for _ in range(len(labels))]
    for i, seq in enumerate(all_split_times):
        diff_vals = np.diff(np.array(seq))
        group_idx = i // seeds
        split_times_escape[group_idx].append(diff_vals)

    split_times_escape_concat = [[] for _ in range(len(labels))]
    for i, group in enumerate(split_times_escape):
        to_plot = np.concatenate(group)
        split_times_escape_concat[i] = to_plot

    if vary == 'N':
        data_flipped  = split_times_escape_concat[::-1]      # swap column order
        labels_flipped = labels[::-1]
        colors_flipped = colors[::-1]
    if vary == 'lambda':
        data_flipped  = split_times_escape_concat     
        labels_flipped = labels
        colors_flipped = colors

    data_flipped.append(all_intervals)
    labels_flipped.append('exp.\ndata')
    colors_flipped.append('#8bc34a')

    # for i in np.arange(len(data_flipped)):
        # print(np.mean(data_flipped[i]))
        # print(np.std(data_flipped[i]))

    fig, axes = joypy.joyplot(
        data_flipped,
        bw_method=0.3,
        labels=labels_flipped,
        figsize=(3.1, 2.7),
        kind="kde",      # plot a kernel density estimate for each column
        overlap=0.7,     # how much the ridges overlap
        fade=True,        # fade overlapping colors slightly
        color=colors_flipped,
        alpha=1
    )
    axes[-1].set_xlabel('Branch intervals (mins)')
    axes[1].set_ylabel('escape rate')
    axes[0].set_title('Timing between branches')

    plt.savefig(f'{name}{panels}ridge.pdf', dpi=300, format='pdf', bbox_inches='tight')
