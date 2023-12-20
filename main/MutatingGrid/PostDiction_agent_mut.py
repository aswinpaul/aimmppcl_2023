#!/usr/bin/env python
# coding: utf-8

# In[1]:


m_trials = 20
n_trials = 600
horizon = 15000
sw = 300

from GridEnv import mutating_gridenvironment
# Environment mutating_gridenvironment.mutating_env()

import numpy as np
import random

random.seed(10)
np.random.seed(10)

from ai_agent_dpefe_fns import normalise_A, normalise_B, action_dist, softmax
import pymdp
from pymdp import utils

import pymdp
from pymdp import utils

from scipy.stats import dirichlet
from matplotlib import pyplot as plt
import time
start_time = time.time()

# Generative model

EPS_VAL = 1e-16 # Negligibleconstant

# Environment known factors
num_states = 472
num_actions = 4


start_state = 430
end_state = 369

# Hidden states
s1_size = num_states
num_states = [s1_size]
num_factors = len(num_states)

# Controls
s1_actions = ['0', '1', '2', '3']
num_controls = [len(s1_actions)]

# Transisition dynamics
# Initialised as flat becuase the agent need to learn the dynamics
B = utils.random_B_matrix(num_states, num_controls)*0 + EPS_VAL

numS = 1
for i in num_states:
    numS *= i
numA = 1
for i in num_controls:
    numA *= i
    
o1_size = s1_size
reward_modes = 2 # -0.5 and 10 (Pos/Neg)

num_obs = [o1_size]
num_modalities = len(num_obs)

# Likelhiood Dynamics
A = utils.random_A_matrix(num_obs, num_states)*0 + EPS_VAL
A[0] = np.identity(s1_size)

A = normalise_A(A, num_states, num_modalities)
B = normalise_B(B, num_states, num_controls)

#Prior over hidden-states (Start state is known)
D = utils.obj_array_zeros(num_states)
D[0][start_state] = 1


## State to decision mapping

# Policy state mapping C
C = utils.obj_array_zeros([1])
C[0] = np.random.rand(numA, numS)*0 + EPS_VAL
C = normalise_A(C, num_states, num_modalities)

# Trials

t_length = np.zeros((m_trials, n_trials))
Gamma_plotting = np.zeros((m_trials, n_trials, 100))

for mt in range(m_trials):
    
    Gamma = np.zeros((horizon, 1))
    for i in range(horizon):
        Gamma[i,0] = 0.9
    
    print(mt)
    
    # Policy state mapping C
    C = utils.obj_array_zeros([1])
    C[0] = np.random.rand(numA, numS)*0 + EPS_VAL
    C = normalise_A(C, num_states, num_modalities)
    
    for trial in range(n_trials):
        
        state = start_state
        qs = D

        for t in range(horizon):

            c_s = np.matmul(np.log(C[0]), qs[0])
            p_d = softmax(c_s)
            
            action = np.random.choice([0,1,2,3], size=None, replace=True, p=p_d)

            n_state, reward = mutating_gridenvironment.mutating_env(state, action, trial, sw)

            obs = utils.obj_array_zeros(num_obs)
            obs[0] = utils.onehot(n_state,num_states)

            qs_prev = qs
            qs = pymdp.inference.update_posterior_states(A, obs, prior = None)

            if(reward == 10):
                for ii in range(t):
                    Gamma[ii] -= 1/(t-ii)

            else:
                Gamma[t] += 0.00001
            
            Gamma = np.clip(Gamma, 0, 1)
                        
            # Learning C
            des = utils.onehot(action,num_actions)
            a = np.reshape(des, (des.shape[0],1))
            b = np.reshape(qs_prev[0], (1,qs_prev[0].shape[0]))
            c_up = np.kron(a,b)
            
            C[0] += (1 - 2*Gamma[t][0])*c_up
            C[0] = np.clip(C[0], EPS_VAL, None)
            C = normalise_A(C, num_states, 1)

            state = n_state

            if(reward == 10):
                break
                
        t_length[mt,trial] = t

with open('Gamma_postdiction.npy', 'wb') as file:
    np.save(file, Gamma_plotting)
    
with open('data_PostDicagent.npy', 'wb') as file:
    np.save(file, t_length)

print("--- %s seconds ---" % (time.time() - start_time))

