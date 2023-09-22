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
from pathlib import Path

path = Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

path = Path(os.getcwd())
module_path = str(path.parent.parent) + '/'
sys.path.append(module_path)

from grid_environments.grid_environment import grid_environment as Env 
env = Env(path = '../grid_environments/grid10.txt', stochastic = True, end_state=37)

# Environment grid_env.grid_environment()

#num_states = env.numS
#num_actions = env.numA

# agent
from pymdp.agent_cl import cl_agent
from pymdp.utils import random_A_matrix, random_B_matrix, obj_array_uniform, norm_dist_obj_arr

# %%

# Generative model
s1_size = env.numS
s1_actions = ['Left', 'Right', 'Up', 'Down']

o1_size = s1_size

# Hidden states
num_states = [s1_size]
num_factors = len(num_states)

# Controls
num_controls = [len(s1_actions)]

#Observations
num_obs = [o1_size]
num_modalities = len(num_obs)

# %%

A = random_A_matrix(num_obs, num_states)
A[0] = np.eye(s1_size)

B = random_B_matrix(num_states, num_controls)

goal_state = env.end_state
C = obj_array_uniform(num_obs)

D = obj_array_uniform(num_states)

# %%

# Trial
m_trials = 100
n_trials = 50
time_horizon = 15000

score_vec = np.zeros((m_trials, n_trials))
gamma_vec = np.zeros((m_trials, n_trials))
frames = []

for mt in range(m_trials):
    print(mt)
    
    N = 50
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
        if(trial%10 == 0):
            end_state = np.random.randint(0,env.numS)
            env = Env(path = '../grid_environments/grid10.txt', 
                      stochastic = True, end_state=end_state)
        
        obs, info = env.reset(seed = mt)
        a.tau = 0
        score = 0
        gamma_vec_list = []
        
        for t in range(time_horizon):
            
            action = a.step([obs], learning = True)
            
            obs, reward, terminated, truncated, info = env.step(int(action[0]))
            score += reward
            
            #if(mt == 0 and trial == 67):
                #frames.append(env.render())
                
            # Learning
            if(truncated):
                a.Gamma[:] = 0.55
                a.update_CL(t) #state-action mapping
                
            if(reward == 10):
                a.update_gamma(risk = -0.55)
                a.update_CL(t) #state-action mapping
            
            gamma_vec_list.append(a.Gamma[0][0])
                                
            #Checking for succesful episode
            if terminated or truncated:
                action  = a.step([obs], learning = True)
                break
            
        score_vec[mt,trial] = score
        gamma_vec[mt,trial] = np.array(gamma_vec_list).min()
        
with open('data_cl.npy', 'wb') as file:
    np.save(file, score_vec)
    
with open('gamma_cl.npy', 'wb') as file:
    np.save(file, gamma_vec)