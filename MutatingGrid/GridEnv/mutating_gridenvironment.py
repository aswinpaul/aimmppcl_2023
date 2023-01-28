from GridEnv import grid_environment_1
from GridEnv import grid_environment_2

def mutating_env(state, action, trial_num, sw):
    
    if(trial_num > sw):
        n_s, rew = grid_environment_1.grid_environment(state, action)
    else:
        n_s, rew = grid_environment_2.grid_environment(state, action)
        
    return n_s, rew