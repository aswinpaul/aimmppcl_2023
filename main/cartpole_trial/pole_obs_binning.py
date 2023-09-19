#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 20:14:25 2023

@author: aswinpaul
"""
import numpy as np

def bin_obs(cp, low, high, bins =  5):
    # x-position of ball
    ranges = np.linspace(low, high, bins)     
    for i in range(bins):
        if(cp <= low):
            return(int(0))
        elif(cp >= high):
            return(int(bins-1))
        elif(cp >= ranges[i] and cp < ranges[i+1]):
            return(int(i))

def state_to_obs(state):
    cp = state[0]
    cv = state[1]
    pa = state[2]
    pv = state[3]

    o1 = bin_obs(cp, low = -1, high = 1, bins = 5)
    o2 = bin_obs(cv, low = -3, high = 3, bins = 5)
    o3 = bin_obs(pa, low = -0.3, high = 0.3, bins = 10)
    o4 = bin_obs(pv, low = -3, high = 3, bins = 10)
    
    return([o1, o2, o3, o4])