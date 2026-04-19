# model_utils.py

import numpy as np
import pickle
from numpy.random import seed

import cell_cycle_model as model              

### Fixed parameters::
INITIAL_DEN = 0.2    # initial nuclear density (N/um)
ALPHA = 2.0          # growth rate of each hyphal tip (um/min)
DT = 0.1             # forward euler timestep
T = 500              # total simulation time (mins)
THRESH = 0.23        # nuclear density threhold (N/um)

### Default parameters:
START_N = 80       # starting number of nuclei
N_TOT = 8000       # total number of nuclei, set large enough to never hit this number for T above
D = 0.0            # noise magnitude
W_MEAN = 120       # mean natural period (natural frequencies from 2pi/normal(w_mean, w_std))
W_STD = 40         # standard deviaton for above
LAMBDA = 0.05      # escape rate for cell cycle checkpoint
BETA =  0.03       # rate for branch formation after density exceeds threshold


####################
### various metrics:
####################

def calculate_density(order_all_t, L):
    '''
    compute nuclear density in units (# nuclei) / (100 micrometers)
    
    L(t): total length of cell (including all branches) at each timestep 
    order_all_t: same length as L, contains indicies of active nuclei at each t
    '''

    # find number of nuclei at each timestep:
    N_time = np.array([len(sub) for sub in order_all_t], dtype=float)

    L = np.asarray(L, dtype=float)

    den = N_time / L         # nuclei per linear µm
    den = den * 100.0        # nuclei per 100 µm

    return den


def fit_line(t, y):

    slope, intercept = np.polyfit(t, y, 1)
    y_fit = slope * t + intercept

    return slope, intercept, y_fit


def calc_cycle_lens(dt, t, theta, start_N, influx_idx):
    '''
    Remove oscillators with unknown cell cycle lengths, 
    then calculate cell cycle lengths
    
    '''

    idx_cycle_not_finished = np.nonzero(theta[-1])[0]
    idx_initial = np.arange(start_N)
    idx_not_started = np.where(np.all(theta == 0, axis=0))

    
    idx_remove = np.concatenate((idx_initial, idx_cycle_not_finished, 
                                 idx_not_started[0], influx_idx))
    
    cycle_finished = np.delete(theta, idx_remove, axis=1)
    cell_cycle_lengths = np.count_nonzero(cycle_finished, axis=0)
    
    birth_times = (cycle_finished!=0).argmax(axis=0)
    
    return birth_times * dt, cell_cycle_lengths * dt 


def order_parameter(phases):
    '''
    takes in an array of phase angles, returns kuramoto order parameter
    '''
    r = abs(sum(complex(np.cos(j),np.sin(j)) for j in phases)/len(phases))
    return r


def calculate_sync(theta, order_all_t, val=1000):
    '''
    compute order parameter
    '''
    all_r = []
    for time, order_i in enumerate(order_all_t):
        order_int = np.array(order_i, dtype=int)
        re_ord = theta[time][order_int]  # get phases of active oscillators at specific t
        
        if len(re_ord) > val: #if this t has more than 1000 oscillators active:
            all_r.append(order_parameter(re_ord)) #then get order parameter
        
    if len(all_r) == 0:
        print('issue: see calculate_sync')
        return np.nan
    
    return np.mean(all_r)


def mean_cycles(dt, t, theta, start_N, influx_idx, birth_cutoff=100.0):
    '''
    Compute mean cycle length for nuclei born before birth_cutoff minutes
    '''
    birth_times, cycle_lens = calc_cycle_lens(dt, t, theta, start_N, influx_idx)
    mask = birth_times < birth_cutoff

    if np.sum(mask) == 0:
        print('see mean_cycles fn')

    return np.mean(cycle_lens[mask])


##############################
### make parameter dictionary:
##############################

def build_parameters(start_N, N, options, omega_mean=W_MEAN, escape_rate=LAMBDA):
    '''
    Build the parameters dict for a single simulation run
    '''

    # initial phases:
    theta0 = np.concatenate((np.random.uniform(0, 2*np.pi, size=start_N),
                             np.zeros(N - start_N)))

    # all natural frequencies:
    omega = np.abs(2 * np.pi / np.random.normal(omega_mean, 
                                                options.get('w_std', W_STD), N))

    # pre determined split_times:
    # (this only does anything if branch times are set to random)
    # intentionally using global W_MEAN value as we don't vary this
    split_times = np.cumsum(np.abs(np.random.normal(W_MEAN, W_STD, 100)))


    params = {
        'dt': DT,
        'T': T,
        'theta0': theta0,
        'omega': omega,
        'D': options.get('D', D),
        'L0': start_N * (1/INITIAL_DEN),
        'initial_den': INITIAL_DEN,
        'growth_rate': ALPHA,
        'split_times': split_times,
        'split_thresh': THRESH,
        'G1_exit_thresh': THRESH,
        'escape_rate': escape_rate,
        'split_rate': options.get('beta', BETA),
    }

    return params


##############################
### run model and save output:
##############################

def run_model(config, save_as=None):
    '''
    Runs simulations 

    Example of expected config:

    config = {
        'seeds': 50,
        'loop_type': 'escape_rate',
        'loop_values': [5, 0.5, 0.05, 0.005],
        'options': {
            ###choose one: ################
            'tip_splits': 'random',
            # 'tip_splits': 'all_random',
            # 'tip_splits': 'triggered',
            ###choose one: #################
            # 'frequencies': 'random',
            'frequencies': 'G1_exit',
            ################################
            # optional, include to override defaults:
            # 'N': 8000,         
            # 'D': 0,            
            # 'w_std': 40,        
            # 'w_mean': 120,      
            # 'lambda': 0.05,    
            # 'beta': 0.03,       
        }
    }

    '''

    seeds = config['seeds']
    loop_type = config['loop_type']
    loop_values = config['loop_values']
    options = config['options']

    N = config.get('N', N_TOT) # total number of nuclei for solution matrix

    num_loops = len(loop_values)

    # store these things in loops below:
    slopes = np.zeros((num_loops, seeds)) # slopes of linear fit to density
    stds = np.zeros((num_loops, seeds))   # standard deviation of detrended density
    sync = np.zeros((num_loops, seeds))   # order parameter
    cycles = np.zeros((num_loops, seeds)) # average cycle length per run
    all_split_times = []                  # array of all branch times per run

    for j, lv in enumerate(loop_values):
        for i in range(seeds):
            seed(i)

            ### if varying initial number of nuclei:
            if loop_type == 'startN':
                start_N = lv
                omega_mean = options.get('w_mean', W_MEAN)
                escape_rate = options.get('lambda', LAMBDA)

            ### if varying checkpoint escape rate (lambda):
            elif loop_type == 'escape_rate':
                start_N = START_N
                omega_mean = options.get('w_mean', W_MEAN)
                escape_rate = lv

            ### if varying checkpoint natural freq mean:
            elif loop_type == 'omega_mean':
                start_N = START_N
                omega_mean = lv
                escape_rate = options.get('lambda', LAMBDA)

            else:
                raise ValueError(f"Unknown loop_type: {loop_type}")

            # build parameters and run the model
            params = build_parameters(
                start_N=start_N,
                N=N,
                options=options,
                omega_mean=omega_mean,
                escape_rate=escape_rate,
            )

            (L, theta, t, order_all_t,
             end_simulation, num_splits_t, split_times,
             influx_idx) = model.solve(start_N, N, params, options)
            
            # Density and trend
            den = calculate_density(order_all_t, L) # units: N/100um
            slope, intercept, y_fit = fit_line(t, den)

            # Store:
            slopes[j, i] = slope             # slope of best fit line to density
            stds[j, i] = np.std(den - y_fit) # detrended density
            sync[j, i] = calculate_sync(theta, order_all_t)
            cycles[j, i] = mean_cycles(params["dt"], t, theta, start_N, influx_idx)
            all_split_times.append(split_times)

    data = {
        'slopes': slopes,
        'stds': stds,
        'sync': sync,
        'cycles': cycles,
        'all_split_times': all_split_times,
        'loop_values': loop_values,
        'loop_type': loop_type,
        'seeds': seeds,
        'N': N,
        'options': options,
    }

    if save_as is not None:
        with open(save_as, "wb") as f:
            pickle.dump(data, f)

    return data


