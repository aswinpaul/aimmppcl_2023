# -*- coding: utf-8 -*-
import numpy as np

m_trials = 99
n_trials = 200
data_dpefe = np.zeros((m_trials, n_trials))
for i in range(m_trials):
    file_name = 'gamma_cl_' + str(i) + '.npy'
    with open(file_name, 'rb') as file:
        data_dpefe[i,:] = np.load(file) 
        
with open('../gamma_cl.npy', 'wb') as file:
    np.save(file, data_dpefe)
