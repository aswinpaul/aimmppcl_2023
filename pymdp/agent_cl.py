#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:41:25 2023
@author: aswinpaul
"""

# This is needed since pymdp si base agent is in a diff folder
import os
import sys
from pathlib import Path

path = Path(os.getcwd())
module_path = str(path.parent.parent) + '/'
sys.path.append(module_path)

# importing the existing classical AI agent in pymdp to reuse inference and learning
from pymdp.agent import Agent
import numpy as np
from pymdp.utils import obj_array_zeros, onehot, norm_dist_obj_arr
from pymdp.maths import softmax

class cl_agent(Agent):
    def __init__(self, A = None, B = None, C = None, D = None, 
                  action_precision = 1,
                  planning_precision = 1,
                  memory_horizon = 1,
                  gamma_initial = 0.55):
        
        super().__init__(A = A, B = B, C = C, D = D,
                         pA = A, pB = B, pD = D,
                         alpha = action_precision,
                         gamma = planning_precision)
        
        # new dimentions of paramters (only for planning)
        self.numS = 1
        self.numA = 1
        for i in self.num_states:
            self.numS *= i
        for i in self.num_controls:
            self.numA *= i
            
        self.melted_ns = [self.numS]
        self.melted_nc = [self.numA]
        self.melted_no = self.num_obs
        
        # CL method variables
        # Policy state mapping CL
        self.EPS_VAL = 1e-16
        self.CL = obj_array_zeros([1])
        self.CL[0] = np.random.rand(self.numA, self.numS)*0 + self.EPS_VAL
        self.CL = norm_dist_obj_arr(self.CL)
        
        # Risk (Gamma)
        self.Gamma = np.zeros((memory_horizon, 1))
        for i in range(memory_horizon):
            self.Gamma[i,0] = gamma_initial
        
        self.memory_horizon = memory_horizon
        self.gamma_initial = gamma_initial
        self.last_n_qs = []
        self.last_n_action = []
        
    def take_decision(self):
        c_s = np.matmul(np.log(self.CL[0]), self.qs_melted[0])
        p_d = softmax(self.alpha*c_s)
        self.q_pi = p_d
        
    def update_gamma(self,terminated = False, risk = 1):
        self.Gamma += risk
        self.Gamma = np.copy(np.clip(self.Gamma, 0, 1))
        
    def update_CL(self, tau):
        if(tau < self.memory_horizon):
            t = tau
        else:
            t = self.memory_horizon
            
            for i in range(t):
                qs_prev = self.last_n_qs[i]
                action = self.last_n_action[i]
                
                # Learning CL
                des = onehot(action, self.numA)
                a = np.reshape(des, (des.shape[0],1))
                b = np.reshape(qs_prev[0], (1,qs_prev[0].shape[0]))
                c_up = np.kron(a,b)
                
                self.CL[0] += (1 - 2*self.Gamma[i][0])*c_up
                self.CL[0] = np.clip(self.CL[0], self.EPS_VAL, None)
                self.CL = norm_dist_obj_arr(self.CL)
                
    def step(self, obs_list, learning = True):
        """
        Agent step combines the following agent functions:
        Combines Inference, Planning, Learning, and decision-making.
        This function represents the agent-environment loop in behaviour where an "environment" feeds observations
        to an "Agent", then the "Agent" responds with actions to control the "environment".
        Usage: agent.step([obs_list])
        Returns: Action(s) from agent to environment
        """
        if(self.tau == 0):
            # Inference
            self.last_n_qs = []
            self.last_n_action = []
            
            self.qs_prev = np.copy(self.D)
            self.infer_states(obs_list)
            
            # New belief for planning
            self.qs_melted = obj_array_zeros(self.melted_ns)
            if self.qs is not None:
                q = 1
            for i in range(len(self.num_states)):
                q = np.kron(q, self.qs[i])
            self.qs_melted[0] = q
            
            self.last_n_qs.append(np.copy(self.qs_melted))
            
            # Learning D
            if(learning == True):
                self.update_D(self.qs)
            
            # Decision making
            self.take_decision()
            self.sample_action()
            
            action = int(self.action[0])
            
            if(len(self.last_n_action) > self.memory_horizon):
                self.last_n_action.pop(0)
            self.last_n_action.append(action)
            
            self.tau += 1

        else:
            # Inference
            self.qs_prev = np.copy(self.qs)
            self.infer_states(obs_list)
            
            # Learning model parameters
            if(learning == True):
                # Updating b
                self.update_B(self.qs_prev)
                # Updating a
                self.update_A(obs_list)
            
            # New belief for planning
            self.qs_melted = obj_array_zeros(self.melted_ns)
            if self.qs is not None:
                q = 1
            for i in range(len(self.num_states)):
                q = np.kron(q, self.qs[i])
            self.qs_melted[0] = q
            
            if(len(self.last_n_qs) > self.memory_horizon):
                self.last_n_qs.pop(0)
            self.last_n_qs.append(np.copy(self.qs_melted))
            
            # Decision making
            self.take_decision()
            self.sample_action()
            
            #Assuming all actions are in first hidden state (to be improved for all hidden states)
            action = int(self.action[0])
            
            if(len(self.last_n_action) > self.memory_horizon):
                self.last_n_action.pop(0)
            self.last_n_action.append(action)
            
            self.tau += 1

        return(self.action)
