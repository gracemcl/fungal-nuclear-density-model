import numpy as np


def division(k, theta, N, which_eqns, order, end_simulation):
    '''
    if any oscillator completed a period, this removes that oscillar and adds two
    new ones.

    order:      Gives spatial order of active oscillators (*see note below*).
                If order was [1,2] and oscillator 2 completed a period,
                order would become [1,3,4]. 
                
    which_eqns: Gives which oscillators are active, array length N. For the example 
                above, which_eqns would go from [0,1,1,0,0,0,0,...N] to 
                [0,1,0,1,1,0,0,...N]. 

    *note*: order is a holdover from another version of the model (that is the focus 
    of a separate manuscript) where we focus on Kuramoto style coupling between a smaller 
    number of oscillators (order is tracking spatial positions in 1D). Kept it here in 
    case we want to extend this model in the future to include spatial information (this would 
    require modeling how nuclei move through the branched network, which is beyond the scope of this
    study). Also use it for plotting

    '''

    # if oscillator i finished a period, period_complete[i] is 1
    period_complete = theta[k+1] >= 2*np.pi 
    
    # if any oscillators have finished a period: 
    if any(period_complete) == 1:    
        for i in np.nonzero(period_complete)[0]:
            idx = np.argwhere(order==i)[0][0]
            order[idx] = np.max(order)+1
            order = np.insert(order, idx+1, np.max(order)+1) 

        which_eqns = np.zeros(N) 
        if np.max(order) < N:
            for i in order:
                which_eqns[int(i)] = 1
            end_simulation = False
        else:
            end_simulation = True

    return which_eqns, order, end_simulation


def influx(k, theta, N, which_eqns, order, end_simulation):
    '''
    Influx of nuclei from region of the cell not modeled (see diagram in 
    the manuscript's supplemental figure S1E)

    As described in the above function, "order" has been kept in the 
    model even though spatial location is not being used. Here we just 
    add incoming nuclei to the end of order.
    
    which_eqns is updated to reflect the new oscillator 
    '''
    
    new_idx = np.max(order) + 1
    order = np.append(order, new_idx)

    if np.max(order) < N:
        which_eqns[new_idx] = 1
        theta[k+1][new_idx] = np.random.uniform(0, 2*np.pi) 

        end_simulation = False

    else:
        end_simulation = True

    return theta, which_eqns, order, end_simulation, new_idx


def update_splits_random(k, t, num_splits, num_splits_t, parameters):
    '''
    For the version of the model where branch intervals are random, but new sets of branches
    form simultaneously.
    Updates num_splits and num_splits_t when the next pre generated split time is reached.
    Split times are drawn in model_utils and passed in via parameters['split_times'].
    (num_splits starts at 1 (immediate split at t=0), so index below is offset by 1)
    '''

    split_times = parameters['split_times']
    if t[k+1] > split_times[num_splits-1]: 
        num_splits += 1
        num_splits_t[k+1:] = num_splits

    return num_splits, num_splits_t


def update_splits_all_random(x, dt, num_tips, parameters):
    '''
    For the version of the model where each individual tip branches at a randomly drawn time,
    independent of all other tips (supplemental figure S2C only).
    Advances each tip's countdown timer by dt. When a timer expires, replaces it with a new
    draw and appends a sibling tip, both drawn from N(120, 40).
    '''

    x = list(map(lambda xi: xi - dt, x))
    for index, elem in enumerate(x):
        if elem <= 0.01:
            x[index] = np.abs(np.random.normal(120, 40))
            x.append(np.abs(np.random.normal(120, 40)))
    num_tips.append(len(x))

    return x, num_tips


def update_splits_triggered(k, dt, L, order, num_splits, num_splits_t, last_split, 
                            split_times, parameters):
    '''
    For the version of the model where a branch is triggered stochastically when nuclear
    density exceeds a threshold. A refractory period (70 min) prevents immediate re-splitting.
    Branch occurs with probability split_rate * dt per timestep.
    '''

    N_curr = len(order)
    L_curr = L[k+1]
    if N_curr/L_curr > parameters['split_thresh'] and (k+1)-last_split > 70/dt:
        p_split = parameters['split_rate'] * dt
        if np.random.random(1) < p_split:
            num_splits += 1
            last_split = k+1
            num_splits_t[k+1:] = num_splits
            split_times.append((k+1)*dt)

    return num_splits, num_splits_t, last_split, split_times


def apply_G1_checkpoint(k, dt, L, order, theta, parameters):
    '''
    When nuclear density exceeds G1_exit_thresh, oscillators that reach phase pi
    (the G1/S boundary) are held there with probability (1-escape_rate*dt).
    '''

    N_curr = len(order)
    L_curr = L[k+1]
    
    if N_curr/L_curr > parameters['G1_exit_thresh']:
        
        G1_exit = np.logical_and(theta[k+1] > np.pi, theta[k] <= np.pi)
        p_exit = parameters['escape_rate'] * dt
        
        rnd = np.random.random(G1_exit.shape)
        
        checkpt_mask = G1_exit & (rnd > p_exit)
        theta[k+1][checkpt_mask] = np.pi

    return theta

           

def dL_dt_fn(num_splits, num_tips, parameters, options):

    '''
    Function giving how the total length of the cell 
    changes with time. 

    options['tip_splits'] == 'all_random' is the case where the time intervals 
    between each individual branch event are randomly chosen. This only applies to 
    supplemental figure S2C. 
    
    All other branching modes ('random' and 'triggered') assume 
    branches emerge in synchronized pairs, so tip count is 2^num_splits, where
    num_splits is the total number of branch events 
  
    '''
    
    growth_rate = parameters['growth_rate']

    if options['tip_splits'] == 'all_random': 
        dL_dt = growth_rate * num_tips 
    else:
        dL_dt = growth_rate * 2**num_splits #there are 2^n total growing tips
    
    return dL_dt


def white_noise_array(N, parameters):
    '''
    white noise array to be added to solution at each time point (length N)
    '''
    D = parameters['D']
    dt = parameters['dt']
    
    noise = np.sqrt(2*D*dt)*np.random.normal(0, 1, size=N)
    return noise


def d_theta_dt_fn(parameters):
        '''
        rhs of ODE
        '''

        d_theta_dt = parameters['omega']

        return d_theta_dt


def forward_euler(num_splits, num_tips, N, dt, parameters, options, k, 
                  t, L, theta, order, which_eqns):
    '''
    Forward euler with white noise 

    see division function above for description of which_eqns
    '''

    theta_current = (theta[k] + 
                     dt*d_theta_dt_fn(parameters) + 
                     white_noise_array(N, parameters))

    theta_current = np.abs(theta_current) # reflecting any negative phases

    L_current = (L[k] + dt*dL_dt_fn(num_splits, num_tips, parameters, options))

    return L_current, theta_current*which_eqns



def setup(start_N, N):
    '''
    Sets up order and which_eqns

    see division function above for description of order and which_eqns 
    '''

    order = np.arange(start_N)
    
    which_eqns = np.zeros(N)
    for i in order:
        which_eqns[int(i)] = 1

    return order, which_eqns



def solve(start_N, N, parameters, options):
    '''
    Solve d_theta/dt=f(theta,t), theta(0)=theta0, with n steps until t=T
    or until the Nth oscillator exists. 

    theta is (T,N) solution matrix. N is the total number of 
    oscillators that the simulation allows for (the total number at any given time varies
    through each simulation). Each oscillator to exist gets a unique index. 
    
    For the paper figures, made sure to set N large enough for each set of simulations
    so that T is the end condition 

    "splits" refers to the cell splitting, i.e. a branch forming
    '''
    
    #######################################################
    #-- general setup: ------------------------------------
    #######################################################

    # see division function above for description of "order".
    # which_eqns gives which oscillators are active
    order, which_eqns = setup(start_N, N)

    dt = parameters['dt']
    n = int(parameters['T']/dt)
    t = np.zeros(n+1)
    t[0] = 0

    theta = np.zeros((n+1, N)) 
    theta[0] = parameters['theta0']
    
    L = np.zeros((n+1, 1))
    # L0 set based on initial N and nuclear density, see model_utils
    L[0] = parameters['L0'] 
    
    # storing the arbitrary order of oscillators, using this for visualizations only:
    order_all_t = [order.copy()] 

    end_simulation = False

    #---- setup for options['tip_splits'] == 'random' or 'triggered': ----

    num_splits = 1 # number of splits so far (starting with an imediate split)
    num_splits_t = np.ones(n+1) # to store number of splits through time
    last_split = 0 # time since last split

    # split_time is to keep track of the times splits happen when they are triggered
    # if options['tip_splits'] == 'random', then will be replaced below
    # by the random times that were pre-generated:
    split_times = [] 
  
    #---- setup for options['tip_splits'] == 'all_random': ----------------
    
    # (case where time intervals between each individual branch event are randomly chosen)
    # all simulations start with an immediate split (so tw initial branches)
    # x: initial countdown timers (min) for each initial tip 
    # (below, when a timer hits 0, that tip branches into two new tips)
    x = [np.abs(np.random.normal(120, 40)), np.abs(np.random.normal(120, 40))]
    num_tips = [2]

    #---- setup for influx: -----------------------------------------------

    # influx_idx is keeping track of indicies of oscillators 
    # that come in from the influx fn. Storing as they need to be ignored
    # when measuring cell cycle lengths (since unknown birth time):
    influx_idx = [] 
    # time interval for oscillators entering simulation via influx fn:
    influx_interval = (1/parameters['initial_den'])/(parameters['growth_rate']) 
    next_influx = influx_interval # initialize timer

    

    #######################################################
    #-- solve: --------------------------------------------
    #######################################################

    for k in np.arange(n):
        t[k+1] = t[k] + dt
        L[k+1], theta[k+1] = forward_euler(num_splits, len(x), N, dt, parameters, options, 
                                           k, t, L, theta, order, which_eqns)
        ### divisions: ##########
        which_eqns, order, end_simulation = division(k, theta, N, which_eqns, order, 
                                                     end_simulation)

        ### influx: ##########
        if ((k+1) * dt) >= next_influx:
            theta, which_eqns, order, end_simulation, new_idx = influx(k, theta, N, which_eqns, 
                                                                       order, end_simulation)
            next_influx += influx_interval
            influx_idx.append(new_idx)

        #------------------
        if end_simulation:
            break
        #------------------

        ### branching: ##########
        if options['tip_splits'] == 'random':
            num_splits, num_splits_t = update_splits_random(k, t, num_splits, num_splits_t, 
                                                            parameters)
        elif options['tip_splits'] == 'all_random':
            x, num_tips = update_splits_all_random(x, dt, num_tips, parameters)

        elif options['tip_splits'] == 'triggered':
            num_splits, num_splits_t, last_split, split_times = update_splits_triggered(
                k, dt, L, order, num_splits, num_splits_t, last_split, split_times, parameters)

        ### cell cycle checkpoint: ####
        if options['frequencies'] == 'G1_exit':
            theta = apply_G1_checkpoint(k, dt, L, order, theta, parameters)


        #--------------------------
        order_all_t.append(order.copy())
        #--------------------------

    influx_idx = np.array(influx_idx, dtype=int)

    if end_simulation: # if reached max N:
        return (np.concatenate(L[0:k]), theta[0:k, 0:np.max(order)+1], t[0:k], 
                order_all_t[0:k], end_simulation, num_splits_t[0:k], split_times, influx_idx)
    else: # if reached max t:
        return (np.concatenate(L), theta[:, 0:np.max(order)+1], t, order_all_t, 
                end_simulation, num_splits_t, split_times, influx_idx)
    
    
    

