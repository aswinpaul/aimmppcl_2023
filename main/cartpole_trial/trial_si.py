#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:48 2022

@author: aswinpaul
"""

import numpy as np

# This is needed agents are in a diff folder
import os
import sys

mtrial = int(sys.argv[1])
print(mtrial)

from pathlib import Path
import gymnasium

eta = 12000 #int(sys.argv[1])

path = Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

env = gymnasium.make("CartPole-v1", render_mode = 'rgb_array')
from pole_obs_binning import state_to_obs
# Environment grid_env.grid_environment()

# agent
from agents.agent_si_z_learning import si_agent_learnc as si_agent
from pymdp.utils import random_A_matrix, random_B_matrix, obj_array_uniform, norm_dist_obj_arr

# %%

# Generative model

s_size = 500

num_states = [s_size]
num_factors = len(num_states)

# Controls
s_actions = ['Right', 'Left']


num_controls = [len(s_actions)]

# Observations

num_obs = num_states
num_modalities = len(num_obs)

# %%

A = random_A_matrix(num_obs, num_states)
A[0][:,:] = np.eye(s_size)
            
    
# %%

B = random_B_matrix(num_states, num_controls)*0 + 1e-16
B = norm_dist_obj_arr(B)

C = obj_array_uniform(num_obs)

D = obj_array_uniform(num_states)

# %%
    
# Trial
n_trials = 200

time_horizon = 10000
score_vec = np.zeros((n_trials))
    
N = 1
a = si_agent(A = A,
             B = B,
             C = C,
             D = D,
             planning_horizon = N,
             eta_par = eta,
             action_precision = 1,
             planning_precision = 1)

a.lr_pB = 1e+100
a.lr_pD = 1
a.lr_PA = 1

for trial in range(n_trials):        
    obs, info = env.reset(seed=mtrial)
    obs_list = state_to_obs(obs)
    o = [obs_list[0]*100 + obs_list[2]*10 + obs_list[3]]
    
    score = 0
    
    cc = a.C[0]
        
    a.tau = 0  
    
    for t in range(time_horizon):

        action = a.step(o)
        
        prev_obs = o
        obs, reward, terminated, truncated, info = env.step(int(action[0]))
        obs_list = state_to_obs(obs)
        o = [obs_list[0]*100 + obs_list[2]*10 + obs_list[3]]
           
        if(terminated):
            r = -1
            a.update_c(prev_obs, obs_list, r, terminal = True)
        else:
            r = 1
            a.update_c(prev_obs, obs_list, r, terminal = False)
        
        score += reward  
        
        #Checking for succesful episode
        if terminated or truncated:
            action  = a.step(o)
            break
        
    score_vec[trial] = score

file_name = 'data_si/si_' + str(mtrial) + '.npy'
with open(file_name, 'wb') as file:
    np.save(file, score_vec)
