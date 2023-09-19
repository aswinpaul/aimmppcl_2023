#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:06:01 2022

@author: aswinpaul
"""

import numpy as np

# This is needed agents are in a diff folder
import os
import sys
from pathlib import Path
import gymnasium
import math

path = Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

env = gymnasium.make("CartPole-v1")
# Environment grid_env.grid_environment()
from pole_obs_binning import state_to_obs

num_states = 10000
num_actions = 2

from agents.agent_dynaq import dynaq_agent as dqa

episodes = 400
seedloops = 100
mem_replay = 10
time_horizon = 15000

#Initialsing the model M(x,a,x',reward)
score_vec = np.zeros((seedloops, episodes))
l = []
for sl in range(seedloops):
    
    print(sl)
    agent = dqa(num_states, num_actions, replay=mem_replay, gamma=1)
        
    # Changing random seeds
    rseed = sl;
    np.random.seed(rseed)
    
    for ts in range(episodes): 
        
        done = False
        tau = 0
        score = 0
        observatio, info = env.reset(seed=sl)
        observation = state_to_obs(observatio)
        obs_int = observation[0]*1000 + observation[1]*100 + observation[2]*10 + observation[3]
        
        for t in range(time_horizon):
            
            tau += 1
            
            l.append(observatio[3])
            action = agent.decision_making(obs_int)
            
            obs_prev = obs_int   
            #Fetching next-state reward from envrionment-function
            observatio, reward, terminated, truncated, info = env.step(action)
            observation = state_to_obs(observatio)
            obs_int = observation[0]*1000 + observation[1]*100 + observation[2]*10 + observation[3]
            
            score += reward
            s = score
            if ts > 200:
                new_theta = 6 * 2 * math.pi / 360
                terminated = True if (observation[2] > new_theta or observation[2] < -new_theta) else terminated
                
            if terminated:
                s = -10
            agent.learning_with_replay(obs_prev, action, obs_int, s)
            
            if terminated or truncated:
                break
    
        score_vec[sl][ts] = score

with open('data_dynaq_10.npy', 'wb') as file:
    np.save(file, score_vec)