m_trials = 10
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

from scipy.stats import dirichlet
from scipy.stats import entropy
from matplotlib import pyplot as plt
import time
start_time = time.time()

# Generative model

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

#Observations
o1_size = s1_size
# reward_modes = 2 # -1 and 10 (Pos/Neg)

num_obs = [o1_size]
num_modalities = len(num_obs)

EPS_VAL = 1e-16 # Negligibleconstant

# Functionals in generative model

# Likelhiood Dynamics
A = utils.random_A_matrix(num_obs, num_states)*0 + EPS_VAL
A[0] = np.identity(s1_size)

# Transisition dynamics
# Initialised as flat becuase the agent need to learn the dynamics
B = utils.random_B_matrix(num_states, num_controls)*0 + EPS_VAL

numS = 1
for i in num_states:
    numS *= i
numA = 1
for i in num_controls:
    numA *= i

A = normalise_A(A, num_states, num_modalities)
B = normalise_B(B, num_states, num_controls)

B_naive = B

# Prior preferences for biasing the generative model to control behaviour
# The preferences are set uniform for all the hidden-states except the reward function
C = utils.obj_array_zeros(num_obs)
C[0][end_state] = 10
C[0] = softmax(10*C[0])

# Policy state mapping C
C_1 = utils.obj_array_zeros([1])
C_1[0] = np.random.rand(numA, numS)*0 + EPS_VAL
C_1 = normalise_A(C_1, num_states, num_modalities)

#Prior over hidden-states (Start state is known)
#Prior over hidden-states (Start state is known)
D = utils.obj_array_zeros(num_states)
D[0][start_state] = 1


# Half planning

T = 25


t_length = np.zeros((m_trials, n_trials))
Beta_plotting = np.zeros((m_trials, n_trials, numS))
Gamma_plotting = np.zeros((m_trials, n_trials, 100))

t_best = 100

for mt in range(m_trials):
    
	Gamma = np.zeros((horizon)) + 0.9
	Beta = np.zeros((numS)) + 0.5

	print(mt)
	B = B_naive
	Q_pi = action_dist(A, B, C, T, sm_par = 1)

	# Policy state mapping C
	C_1 = utils.obj_array_zeros([1])
	C_1[0] = np.random.rand(numA, numS)*0 + EPS_VAL
	C_1 = normalise_A(C_1, num_states, num_modalities)

	for trial in range(n_trials):
		Beta_plotting[mt, trial, :] = Beta
		print(mt, "trial: ", trial)
		
		if(trial % 10 == 0):
			if(t_length[mt, trial-1] > t_best): 
				Q_pi = action_dist(A, B, C, T, sm_par = 1)

		state = start_state
		qs = D
		qs_prev = qs

		for t in range(horizon):
		    
			c_s = np.log(np.matmul(C_1[0], qs[0]))
			p_d_1 = softmax(c_s)

			p_d_2 = np.matmul(Q_pi[:,:], qs[0])
					    
			ent_1 = entropy(p_d_1)
			ent_2 = entropy(p_d_2)

			Beta[:] += qs[0] * (ent_1 - ent_2)
			Beta = np.clip(Beta, 0, 1)
				
			bias = np.matmul(Beta[:], qs[0])

			p = (1 - bias)*np.log(p_d_1) + bias*np.log(p_d_2)            
			p = softmax(p)

			action = np.random.choice([0,1,2,3], size=None, replace=True, p = p)
			n_state, reward = mutating_gridenvironment.mutating_env(state, action, trial, sw)

			obs = utils.obj_array_zeros(num_obs)
			obs[0] = utils.onehot(n_state,num_states)

			qs_prev = qs
			qs = pymdp.inference.update_posterior_states(A, obs, prior = None)

			#Learning B
			actions = np.array([int(action)])
			B = pymdp.learning.update_state_likelihood_dirichlet(B, B, actions, qs, qs_prev, lr=1.0, factors='all')

			if(reward == 10):
				for ii in range(t):
					Gamma[ii] -= 1/(t-ii)

			else:
				Gamma[t] += 0.00001

			Gamma = np.clip(Gamma, 0, 1)
			if(t < 100):
				Gamma_plotting[mt, trial, t] = Gamma[t]
				
			des = utils.onehot(action,num_actions)
			a = np.reshape(des, (des.shape[0],1))
			b = np.reshape(qs_prev[0], (1,qs_prev[0].shape[0]))
			c_up = np.kron(a,b)

			C_1[0] += (1 - 2*Gamma[t])*c_up
			C_1[0] = np.clip(C_1[0], EPS_VAL, None)
			C_1 = normalise_A(C_1, num_states, 1)

			state = n_state
			if(reward == 10):
				break
			
		t_length[mt,trial] = t
		if(t < t_best):
			t_best = t
		B = normalise_B(B, num_states, num_controls)
        
with open('data_Combinedmodel_half.npy', 'wb') as file:
    np.save(file, t_length)
    
with open('Beta_half.npy', 'wb') as file:
    np.save(file, Beta_plotting)
with open('Gamma_half.npy', 'wb') as file:
    np.save(file, Gamma_plotting)

print("--- %s seconds ---" % (time.time() - start_time))

