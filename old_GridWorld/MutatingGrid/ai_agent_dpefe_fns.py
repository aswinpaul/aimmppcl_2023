import numpy as np
import math
import random
import pymdp
from pymdp.maths import spm_log_single as log_stable
from pymdp import utils
from scipy.stats import dirichlet

EPS_VAL = 1e-16 #negligibleconstant

random.seed(123)
np.random.seed(123)

def softmax(dist):
    """ 
    Computes the softmax function on a set of values
    """

    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output

def log_stable(arr):
    """
    Adds small epsilon value to an array before natural logging it
    """
    return np.log(arr + EPS_VAL)

def kl_div(P,Q):
    n=len(P)
    for i in range(n):
        if(P[i]==0):
            P[i]+=EPS_VAL
        if(Q[i]==0):
            Q[i]+=EPS_VAL
            
    dkl=0
    for i in range(n):
        dkl+=(P[i]*math.log(P[i]))-(P[i]*math.log(Q[i]))
    return(dkl)

def entropy(A):
    """ Compute the entropy of a set of condition distributions, i.e. one entropy value per column """
    
    H_A = - (A * log_stable(A)).sum(axis=0)
    return H_A

def normalise_A(A, num_states, num_modalities):
    for j in range(num_modalities):
        A[j] = A[j] / A[j].sum(axis=0)[np.newaxis,:]
    return A

def normalise_B(B, num_states, num_controls):
    for i in range(len(num_states)):
        for j in range(num_states[i]):
            for k in range(num_controls[i]):
                B[i][:,j,k]=dirichlet.mean(B[i][:,j,k])
            
    return B

#Dynamic programming in G (expected free energy)

def action_dist(A, B, C, T, sm_par):
    
    num_modalities = A.shape[0]
    num_factors = B.shape[0]

    num_states = []
    for i in range(num_factors):
        num_states.append(B[i].shape[0])

    num_obs = []
    for i in range(num_modalities):
        num_obs.append(A[i].shape[0])

    num_controls = []
    for i in range(num_factors):
        num_controls.append(B[i].shape[2])

    numS = 1
    for i in num_states:
        numS *= i
    numA = 1
    for i in num_controls:
        numA *= i

    new_num_states = [numS]
    new_num_controls = [numA]

    new_A = utils.random_A_matrix(num_obs, new_num_states) #* 0 + EPS_VAL
    new_B = utils.random_B_matrix(1, 1) #* 0 + EPS_VAL

    for i in range(num_modalities):
        new_A[i] = np.reshape(A[i], [A[i].shape[0], numS])

    for i in range(num_factors):
        new_B[0] = np.kron(new_B[0],B[i])

    #Expected free energy (Only RISK)
    
    G = np.zeros((T-1,numA, numS))
    Q_actions = np.zeros((T-1,numA, numS))
    Q_pi = np.zeros((numA, numS))

    for mod in range(num_modalities):

        Q_po = np.zeros((A[mod].shape[0], numS, numA))

        for i in range(numS):
            for j in range(numA):
                Q_po[:,i,j] = new_A[mod].dot(new_B[0][:,i,j])

        for k in range(T-2,-1,-1):
            for i in range(numA):

                    if(k==T-2):
                        for j in range(numS):
                            G[k,i,j] += kl_div(Q_po[:,j,i],C[mod]) + np.dot(new_B[0][:,j,i],entropy(new_A[mod]))

                    else:
                        for j in range(numS):
                            G[k,i,j] += kl_div(Q_po[:,j,i],C[mod]) + np.dot(new_B[0][:,j,i],entropy(new_A[mod]))
                        
                            for jj in range(numS):
                                for kk in range(numA):
                                    G[k,i,j] += Q_actions[k+1,kk,jj] * new_B[0][jj,j,i] * G[k+1,kk,jj] 

            #Distribution for action-selection
            for l in range(numS):
                Q_actions[k,:,l] = softmax(sm_par*(-1*G[k,:,l]))
                
    for i in range(numS):
        Q_pi[:,i] = softmax(sm_par*(-1*G[k,:,i]))
                
    return Q_pi
