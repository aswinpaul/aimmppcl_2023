# -*- coding: utf-8 -*-
import numpy as np

m_trials = 99
n_trials = 400
data_dpefe = []

for i in range(m_trials):
    file_name = 'data_dpefe_' + str(i) + '.npy'
    try:
        with open(file_name, 'rb') as file:
            data_dpefe.append(np.load(file))
    except:
        print("Not avail")
         
with open('../data_dpefe_20.npy', 'wb') as file:
    np.save(file, np.array(data_dpefe))