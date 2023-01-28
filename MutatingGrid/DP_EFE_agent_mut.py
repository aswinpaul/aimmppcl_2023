m_trials = 10
n_trials = 600
time_horizon = 15000
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
from matplotlib import pyplot as plt
import time
start_time = time.time()

# Generative model

# Environment known factors
num_states = 472
num_actions = 4

start_state = 430
end_state = 369

# Generative model

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

#Prior over hidden-states (Start state is known)
D = utils.obj_array_zeros(num_states)
D[0][start_state] = 1


# Trial

# Planning horizon
T = 50

t_length = np.zeros((m_trials, n_trials))

t_best = 100

for mt in range(m_trials):
    
	print(mt)
	B = B_naive
	Q_pi = action_dist(A, B, C, T, sm_par = 1)

	for trial in range(n_trials):
		print(mt, "trial: ", trial)
		if(trial % 10 == 0):
			if(t_length[mt, trial-1] > t_best):
				Q_pi = action_dist(A, B, C, T, sm_par = 1)

		state = start_state
		qs = D
		qs_prev = qs

		for t in range(time_horizon):
		    
			action = np.random.choice([0,1,2,3], size = None, replace = True, p = np.matmul(Q_pi, qs[0]))
			n_state, reward = mutating_gridenvironment.mutating_env(state, action, trial, sw)
			state = n_state

			obs = utils.obj_array_zeros(num_obs)
			obs[0] = utils.onehot(state,num_states)

			qs_prev = qs
			qs = pymdp.inference.update_posterior_states(A, obs, prior = None)

			#Learning A
			#A = pymdp.learning.update_obs_likelihood_dirichlet(A, A, obs, qs, lr=1.0, modalities='all')

			#Learning B
			actions = np.array([int(action)])
			B = pymdp.learning.update_state_likelihood_dirichlet(B, B, actions, qs, qs_prev, lr = 1.0, factors='all')

			#Checking for succesful episode
			if(reward == 10):
				break

		t_length[mt,trial] = t
		B = normalise_B(B, num_states, num_controls)

		if(t < t_best):
			t_best = t


with open('data_DynProgagent.npy', 'wb') as file:
    np.save(file, t_length)

# Plotting

print("--- %s seconds ---" % (time.time() - start_time))
