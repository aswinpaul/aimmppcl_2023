#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:53:50 2023

@author: aswinpaul
"""

import numpy as np
import matplotlib.pyplot as plt

data = {}

with open('data_random.npy', 'rb') as file:
    data[0] = np.load(file)
    
with open('data_dynaq_10.npy', 'rb') as file:
    data[1] = np.load(file)

with open('data_cl_5.npy', 'rb') as file:
    data[2] = np.load(file)
    
with open('data_si.npy', 'rb') as file:
    data[3] = np.load(file)
    
with open('data_dpefe_5.npy', 'rb') as file:
    data[4] = np.load(file)

with open('data_dpefe_20.npy', 'rb') as file:
    data[5] = np.load(file)
    
agents = 6
episodes = 200

sample = np.shape(data[0][:,0:episodes][0])[0]

data_mean = {}  
for i in range(agents):
    data_mean[i] = np.mean(np.transpose(data[i][:,0:episodes]), axis=1)
    plt.plot(range(sample-1),data_mean[i][:-1])

data_std = {}    
for i in range(agents):
    fact = 1 if(i == 0 or i == 2) else 1
    data_std[i] = np.std(np.transpose(data[i][:,0:episodes]), axis=1)
    plt.fill_between(range(sample-1), 
                     data_mean[i][:-1] + fact*data_std[i][:-1],
                     data_mean[i][:-1] - fact*data_std[i][:-1],
                     alpha=0.3)

plt.legend(["Random agent", "Dyna-Q agent (memory replay = 10)", "CL method agent (T = 1)", "AIF agent (T = 1)", "DPEFE agent (T = 5)", "DPEFE agent (T = 20)"])

plt.title("Cart Pole - v1 (OpenAI Gym)")

plt.xlabel("Episode number")
plt.ylim(0, None)
plt.xlim(0, episodes)
plt.ylabel("Total score")
plt.savefig('perf_1.png', dpi=500, bbox_inches='tight');
