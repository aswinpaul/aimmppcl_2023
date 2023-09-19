#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:38:48 2022

@author: aswinpaul
"""

# This is needed agents are in a diff folder
import os
import sys
from pathlib import Path
import gymnasium

path = Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

env = gymnasium.make("CartPole-v1", render_mode = "human")
# Environment grid_env.grid_environment()

import numpy as np
np.random.seed(10)

# Trial
m_trials = 100
n_trials = 200
time_horizon = 15000

score_vec = np.zeros((m_trials, n_trials))
obs_save = []

for mt in range(m_trials):
    print(mt)
    
    
    for trial in range(n_trials):
        
        #print(mt, "trial: ", trial)
            
        obs, info = env.reset(seed=mt)
        obs_list = [obs]
        prev_obs_list = obs_list
        
        score = 0
        st = []
        st.append(obs)
        
        for t in range(time_horizon):
            env.render()
            action  = np.random.choice([0,1], size=None, replace=True, 
                                       p=[0.5, 0.5])
            obs, reward, terminated, truncated, info = env.step(action)
            st.append(obs)
            
            prev_obs_list = obs_list
            obs_list = [obs]
            
            score += reward  
            
            #Checking for succesful episode
            if terminated or truncated:
                break

        score_vec[mt,trial] = score
        
with open('data_random.npy', 'wb') as file:
    np.save(file, score_vec)