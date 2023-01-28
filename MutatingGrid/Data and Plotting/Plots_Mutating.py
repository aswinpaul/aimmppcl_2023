#!/usr/bin/env python
# coding: utf-8

# # Plots

# In[1]:


import numpy as np
from matplotlib import pyplot as plt


# %%



def plot_subresultsci(ql,sample,label):
    ql_rewards_means = np.mean(ql, axis=1)
    ql_rewards_stddev = np.std(ql, axis=1)
    
    plt.plot(range(sample-1),ql_rewards_means[:-1], label=label)
    plt.fill_between(range(sample-1), ql_rewards_means[:-1] + ql_rewards_stddev[:-1], ql_rewards_means[:-1] - ql_rewards_stddev[:-1], alpha=0.3)
    plt.ylim([0,None])
    plt.xlabel("Episode number")
    plt.axvline(300, linestyle='--', color='grey')
    plt.legend()


# # DP_EFE Agent vs PostDiction agent

# %%:


# Plotting
with open('data_DynProgagent.npy', 'rb') as file:
    t_length_1 = np.load(file)
    
with open('data_PostDicagent.npy', 'rb') as file:
    t_length_2 = np.load(file)

plot1 = plt.figure(1)
plot_subresultsci(np.transpose(t_length_1),np.shape(t_length_1[0])[0], label="DP-EFE agent")
plot_subresultsci(np.transpose(t_length_2),np.shape(t_length_2[0])[0], label='CL method agent')
plt.title("DP-EFE agent vs CL method agent")
plt.ylabel("Length of episode")
plt.savefig('dpefe_vs_cl_mut.png', dpi=500, bbox_inches='tight', format = 'png')


# # Combined model

# %%


with open('data_Combinedmodel_zero.npy', 'rb') as file:
    t_length_1 = np.load(file)
    
with open('data_Combinedmodel_half.npy', 'rb') as file:
    t_length_2 = np.load(file)
    
with open('data_Combinedmodel_full.npy', 'rb') as file:
    t_length_3 = np.load(file)

plot1 = plt.figure(1)
plot_subresultsci(np.transpose(t_length_1),np.shape(t_length_1[0])[0], label="Planning depth N = 5")
plot_subresultsci(np.transpose(t_length_2),np.shape(t_length_2[0])[0], label="Planning depth N = 25")
plot_subresultsci(np.transpose(t_length_3),np.shape(t_length_3[0])[0], label="Planning depth N = 50")

plt.title("Performance of 'Mixed-model' agent")
plt.ylabel("Length of episode")
plt.savefig('CombModel_mut.png', dpi=500, bbox_inches='tight', format = 'png')


# ### Beta

# %%

with open('Beta_zero.npy', 'rb') as file:
    Beta_zero = np.load(file)
with open('Beta_half.npy', 'rb') as file:
    Beta_half = np.load(file)
with open('Beta_full.npy', 'rb') as file:
    Beta_full = np.load(file)
    
plot1 = plt.figure(1)
plot_subresultsci(np.transpose(np.average(Beta_zero, axis = 2)), np.shape(Beta_zero[0])[0], label="Planning depth N = 5")
plot_subresultsci(np.transpose(np.average(Beta_half, axis = 2)), np.shape(Beta_half[0])[0], label="Planning depth N = 25")
plot_subresultsci(np.transpose(np.average(Beta_full, axis = 2)), np.shape(Beta_full[0])[0], label="Planning depth N = 50")
plt.ylabel("Beta (average over all states)")
plt.title("Beta (Measure of model mixing) in 'Mixed-model' agent")
plt.savefig('Beta_mut.png', dpi=500, bbox_inches='tight', format = 'png')


# ### Gamma

# %%


with open('Gamma_zero.npy', 'rb') as file:
    Gamma_zero = np.load(file)
    
with open('Gamma_half.npy', 'rb') as file:
    Gamma_half = np.load(file)
    
with open('Gamma_full.npy', 'rb') as file:
    Gamma_full = np.load(file)

plot1 = plt.figure(1)
plot_subresultsci(np.transpose(np.average(Gamma_zero, axis = 2)), np.shape(Gamma_zero[0])[0], label="Planning depth N = 5")
plot_subresultsci(np.transpose(np.average(Gamma_half, axis = 2)), np.shape(Gamma_zero[0])[0], label="Planning depth N = 25")
plot_subresultsci(np.transpose(np.average(Gamma_full, axis = 2)), np.shape(Gamma_zero[0])[0], label="Planning depth N = 50")

plt.ylabel("Gamma (start of a trial)")
plt.title("Gamma (Risk) in 'Mixed-model' agent")
plt.savefig('Gamma_mut.png', dpi=500, bbox_inches='tight', format = 'png')

