#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:48 2022

@author: aswinpaul
"""

import numpy as np
np.random.seed(10)

# This is needed agents are in a diff folder
import os
import sys

mtrial = int(sys.argv[1])
print(mtrial)

from pathlib import Path
import gymnasium

path = Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

env = gymnasium.make("CartPole-v1", render_mode = 'rgb_array')
from pole_obs_binning import state_to_obs
# Environment grid_env.grid_environment()

# agent
from agents.agent_dpefe_z_learning import dpefe_agent_z
from pymdp.utils import random_A_matrix, random_B_matrix, obj_array_uniform, norm_dist_obj_arr

# %%

# Generative model

# (Hidden)Factors
# Cart position (Hypothesis)
s1_size = 500

num_states = [s1_size]
num_factors = len(num_states)

# Controls
s1_actions = ['Right', 'Left']

num_controls = [len(s1_actions)]

num_obs = num_states
num_modalities = len(num_obs)

# %%

A = random_A_matrix(num_obs, num_states)
A[0][:,:] = np.eye(s1_size)
        
B = random_B_matrix(num_states, num_controls)*0 + 1e-16
B = norm_dist_obj_arr(B)

C = obj_array_uniform(num_obs)

D = obj_array_uniform(num_states)
        
# %%

# Trial
n_trials = 200
time_horizon = 15000

score_vec = np.zeros((n_trials))

    
N = 20
a = dpefe_agent_z(A = A,
                    B = B,
                    C = C,
                    D = D,
                    planning_horizon = N,
                    action_precision = 1,
                    planning_precision = 1)

a.lr_pB = 1000
a.lr_pA = 1
a.lr_pD = 1

a.trial_tau = 0

for trial in range(n_trials):

    cc = a.C[0]

    obs, info = env.reset(seed=mtrial)
    obs_list = state_to_obs(obs)
    o = [obs_list[0]*100 + obs_list[2]*10 + obs_list[3]]

    a.tau = 0
    score = 0

    for t in range(time_horizon):

        action  = a.step(o)

        prev_obs = o
        obs, reward, terminated, truncated, info = env.step(int(action[0]))
        obs_list = state_to_obs(obs)
        o = [obs_list[0]*100 + obs_list[2]*10 + obs_list[3]]

        #Learning a general C to aid tree-search in soph.inf
        #if reward is not 10 it is a usual step with no additional info
        #if reward is 10, it is a terminal state i.e the goal state
        if(terminated):
            r = -1
            a.update_c(prev_obs, obs_list, reward = r, terminal = True)
        else:
            r = 1
            a.update_c(prev_obs, obs_list, reward = r, terminal = False)

        score += reward

        #Checking for succesful episode
        if terminated or truncated:
            action = a.step(o)
            break

    score_vec[trial] = score

file_name = 'data_dpefe/data_dpefe_' + str(mtrial) + '.npy'
with open(file_name, 'wb') as file:
    np.save(file, score_vec)
