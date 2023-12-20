#Importing needed modules
import sys
import os
import numpy as np
import random
import math

mdppath = './GridEnv/mdp_2.txt'

#storing the file as strings line by line
mdpdata=[]

#Saving arm true means to the array-band (indices indicates arms)
mdp = open(str(mdppath), "r")
for x in mdp:
    mdpdata.append(x)
mdp.close()
#Closing mdp file.

#Determining the nature of mdp
mdptype1=[]
for word in mdpdata[-2].split():
    try:
        mdptype1.append(str(word))
    except (ValueError, IndexError):
        pass
mdptype=mdptype1[1]

#Discount factor
gamma1=[]
for word in mdpdata[-1].split():
    try:
        gamma1.append(float(word))
    except (ValueError, IndexError):
        pass

gamma=float(gamma1[0])

#Number of states
states=[]
for word in mdpdata[0].split():
    try:
        states.append(int(word))
    except (ValueError, IndexError):
        pass
numS=int(states[0])
#Number of actions
actions=[]
for word in mdpdata[1].split():
    try:
        actions.append(int(word))
    except (ValueError, IndexError):
        pass
numA=int(actions[0])
#Start state
start=[]
for word in mdpdata[2].split():
    try:
        start.append(int(word))
    except (ValueError, IndexError):
        pass
startS=int(start[0])
#Terminal states for episodic mdps
if(mdptype=='episodic'):
    terminal=[]
    for word in mdpdata[3].split():
        try:
            terminal.append(int(word))
        except (ValueError, IndexError):
            pass
    no_of_termS=len(terminal)
    terS=terminal

#T-matrix dimensions numS*numS*numA
#R-matrix dimensions numS*numA*numS

T=np.zeros((numS,numA,numS))
R=np.zeros((numS,numA,numS))
if(mdptype=='episodic'):
    for i in range(len(terS)):
        T[terS[i],:,terS[i]]=1

for i in range(4,len(mdpdata)-2):
    trans=[]
    for word in mdpdata[i].split():
        try:
            trans.append(float(word))
        except (ValueError, IndexError):
            pass
    trans
    s1=int(trans[0])
    ac=int(trans[1])
    s2=int(trans[2])
    r=float(trans[3])
    p=float(trans[4])
    T[s1,ac,s2]=p
    R[s1,ac,s2]=r

#useful variables
#numS,numA,startS,terS,mdptype,gamma,T,R

def grid_environment(state, action):
    n_s = np.argmax(T[state, action, :])
    reward = R[state, action, n_s]
    
    return n_s, reward

def get_trueB():
    return T