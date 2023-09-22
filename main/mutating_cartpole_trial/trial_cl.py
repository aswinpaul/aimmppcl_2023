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
import math

path = Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

path = Path(os.getcwd())
module_path = str(path.parent.parent) + '/'
sys.path.append(module_path)

env = gymnasium.make("CartPole-v1", render_mode="rgb_array")
from pole_obs_binning import state_to_obs

# Environment grid_env.grid_environment()

#num_states = env.numS
#num_actions = env.numA

# agent
from pymdp.agent_cl import cl_agent
from pymdp.utils import random_A_matrix, random_B_matrix, obj_array_uniform, norm_dist_obj_arr

# %%

# Generative model

# (Hidden)Factors
# Cart position (Hypothesis)
s1_size = 5

# Cart velocity (Hypothesis)
s2_size = 5

# Pole position (Hypothesis)
s3_size = 10

# Pole velocity (Hypothesis)
s4_size = 10

num_states = [s1_size, s2_size, s3_size, s4_size]
num_factors = len(num_states)

# Controls
s1_actions = ['Right', 'Left']
s2_actions = ['Do nothing']
s3_actions = ['Do nothing']
s4_actions = ['Do nothing']

num_controls = [len(s1_actions), len(s2_actions), len(s3_actions), len(s4_actions)]

num_obs = num_states
num_modalities = len(num_obs)

# %%

A = random_A_matrix(num_obs, num_states)

for i in range(s2_size):
    for j in range(s3_size):
        for k in range(s4_size):
            A[0][:,:,i,j,k] = np.eye(s1_size)
        
for i in range(s1_size):
    for j in range(s3_size):
        for k in range(s4_size):
            A[1][:,i,:,j,k] = np.eye(s2_size)
            
for i in range(s1_size):
    for j in range(s2_size):
        for k in range(s4_size):
            A[2][:,i,j,:,k] = np.eye(s3_size)
            
for i in range(s2_size):
    for j in range(s2_size):
        for k in range(s3_size):
            A[3][:,i,j,k,:] = np.eye(s4_size)
            
B = random_B_matrix(num_states, num_controls)*0 + 1e-16
B = norm_dist_obj_arr(B)

C = obj_array_uniform(num_obs)

D = obj_array_uniform(num_states)

# %%

# Trial
m_trials = 100
n_trials = 200
time_horizon = 15000

score_vec = np.zeros((n_trials))
gamma_vec = np.zeros((n_trials))
frames = []

    
N = 5
a = cl_agent(A = A,
                B = B,
                C = C,
                D = D,
                memory_horizon = N,
                action_precision = 1,
                gamma_initial = 0.55)

a.lr_pB = 1000
a.lr_pA = 1
a.lr_pD = 1

for trial in range(n_trials):
    obs, info = env.reset(seed=mtrial)
    a.tau = 0
    score = 0
    gamma_vec_list = []

    for t in range(time_horizon):

        obs_list = state_to_obs(obs)
        action = a.step(obs_list, learning = True)

        obs, reward, terminated, truncated, info = env.step(int(action[0]))
        score += reward

        if trial > 100:
            new_theta = 6 * 2 * math.pi / 360
            terminated = True if (obs[2] > new_theta or obs[2] < -new_theta) else terminated

        # Learning
        if(terminated):
            a.Gamma[:] = 0.55
            a.update_CL(t) #state-action mapping

        if(score%100 == 0):
            a.update_gamma(risk = -0.2)
            a.update_CL(t) #state-action mapping

        #Checking for succesful episode
        if terminated or truncated:
            action  = a.step(obs_list, learning = True)
            break


        gamma_vec_list.append(np.mean(a.Gamma))

    score_vec[trial] = score
    gamma_vec[trial] = np.array(gamma_vec_list).min()

file_name = 'data_cl/data_cl_' + str(mtrial) + '.npy'
with open(file_name, 'wb') as file:
    np.save(file, score_vec)

file_name = 'data_gamma_cl/gamma_cl_' + str(mtrial) + '.npy'
with open(file_name, 'wb') as file:
    np.save(file, gamma_vec)
