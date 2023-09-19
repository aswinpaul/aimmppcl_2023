#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 15:56:11 2022

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
from pymdp.maths import softmax, kl_div, entropy, softmax_obj_arr
from pymdp.utils import random_A_matrix, random_B_matrix, obj_array_zeros


class dpefe_agent(Agent):
    """
    # Necessary parameters for SI agent

    # num_states
    # num_obs
    # num_controls

    # Optional parameters
    # planning_horizon (default value 1)
    # A = prior for likelihood A (same structure as pymdp.utils.random_A_matrix(num_obs, num_states))
    # B = prior for transisiton matrix B (same structure as pymdp.utils.random_B_matrix(num_obs, num_states))
    # C = prior for preference dist. C (same structure as pymdp.utils.obj_array_zeros(num_obs))
    # D = 0 prior of hidden-state
    # action_precision (precision for softmax in taking decisions) default value: 1
    # planning_precision (precision for softmax during tree search) default value: 1
    # search_threshold = 1/16 parameter pruning tree search in SI tree-search (default value 1/16)

    # Useful combination functions 
    # agent.step([obs_list], learning = False): 
    Combines Inference, planning, learning, and decision-making
    Generative model will be learned and updated over time if learning = True
    """
    def __init__(self, A = None, B = None, C = None, D = None, 
                 planning_horizon = 1,
                 action_precision = 1,
                 planning_precision = 1):
        
        super().__init__(A = A, B = B, C = C, D = D,
                         pA = A, pB = B, pD = D,
                         alpha = action_precision,
                         gamma = planning_precision)
        
        # Normalising prior preference as a distribution over modalities
        self.C = softmax_obj_arr(C)

        self.N = planning_horizon
        self.EPS_VAL = 1e-16
        self.tau = 0
        
    # Melting hidden state factors as single hidden state modality to use locally for planning
    def melting_factors_for_planning(self):
        # Melting here refers to numerically transforming multiple hidden state factors 
        # as one hidden state factor using tensor products
        
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
        
        # New A-matrix for planning
        self.A_melted = random_A_matrix(self.melted_no, self.melted_ns)
        if self.A is not None:
            for i in range(len(self.num_obs)):
                self.A_melted[i] = self.A[i].reshape(self.melted_no[i], self.numS)
        
        # New B-matrix for planning
        self.B_melted = random_B_matrix(self.melted_ns, self.melted_nc)
        if self.B is not None:
            bb = 1
            for i in range(len(self.num_states)):
                bb = np.kron(bb, self.B[i])
            self.B_melted[0] = bb
            
        # New belief for planning
        self.qs_melted = obj_array_zeros(self.melted_ns)
        if self.qs is not None:
            q = 1
            for i in range(len(self.num_states)):
                q = np.kron(q, self.qs[i])
            self.qs_melted[0] = q
        
   # Planning with dynamic programming
    def plan_using_dynprog(self, modalities = False):
        
        self.melting_factors_for_planning()
        self.N = 2 if self.N == 1 else self.N
        
        self.G = np.zeros((self.N-1, self.numA, self.numS)) + self.EPS_VAL
        self.Q_actions = np.zeros((self.N-1, self.numA, self.numS)) + 1/self.numA
        
        T = self.N
        if(modalities == False):
            moda = list(range(self.num_modalities))
        else:
            moda = modalities
        
        for mod in moda:
            Q_po = np.zeros((self.A_melted[mod].shape[0], self.numS, self.numA))
            
            for i in range(self.numS):
                for j in range(self.numA):
                    Q_po[:,i,j] = self.A_melted[mod].dot(self.B_melted[0][:,i,j])
    
            for k in range(T-2,-1,-1):
                for i in range(self.numA):
                    for j in range(self.numS):
                        if(k == T-2):
                            self.G[k,i,j] += kl_div(Q_po[:,j,i],self.C[mod]) + np.dot(
                                self.B_melted[0][:,j,i],entropy(self.A_melted[mod]))
                        else:
                            self.G[k,i,j] += kl_div(Q_po[:,j,i],self.C[mod]) + np.dot(
                                self.B_melted[0][:,j,i],entropy(self.A_melted[mod]))
                            
                            # Dynamic programming backwards in time
                            self.G[k,i,j] += np.sum(np.matmul(np.reshape(np.multiply(
                                self.Q_actions[k+1,:,:],self.G[k+1,:,:]), 
                                (self.numA,self.numS)),np.reshape(self.B_melted[0][:,j,i], 
                                                                  (self.numS,1))))
                        
                # Distribution for action-selection
                for l in range(self.numS):
                    self.Q_actions[k,:,l] = softmax(-1*self.gamma*self.G[k,:,l])
                    
        # Action distribution (assuming current time is time step zero of planning)
        self.q_pi = softmax(-1*self.alpha*np.matmul(self.G[0,:,:], self.qs_melted[0]))  
        
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
            self.infer_states(obs_list)

            # Learning D
            if(learning == True):
                self.update_D(self.qs)
            
            # Planning
            self.plan_using_dynprog()
            
            # Decision making
            self.sample_action()
            
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
                
            # Planning
            self.plan_using_dynprog()
            
            # Decision making
            self.sample_action()
            self.tau += 1

        return(self.action)