###############################################################################
#
#                                   Libraries
#
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sys import stdout
import time

os.chdir('..')

###############################################################################
#
#                                   Methods
#
###############################################################################

# Generates a random matrix of a given size
def initialize_spin_configuration():
    state = np.random.randint(2, size=(state_size,state_size), dtype = bool)
    return state

def b2s(b):
    s = b.astype(int)*2-1
    return(s)

def displace_periodic_matrix(matrix):
    matrix_down = np.zeros(matrix.shape)
    matrix_down[1:,:] = matrix[:-1,:]
    matrix_down[0,:] = matrix[-1,:]
    matrix_down = matrix_down[np.newaxis,:]
    #
    matrix_up = np.zeros(matrix.shape)
    matrix_up[:-1,:] = matrix[1:,:]
    matrix_up[-1,:] = matrix[0,:]
    matrix_up = matrix_up[np.newaxis,:]
    #
    matrix_left = np.zeros(matrix.shape)
    matrix_left[:,:-1] = matrix[:,1:]
    matrix_left[:,-1] = matrix[:,0]
    matrix_left = matrix_left[np.newaxis,:]
    #
    matrix_right = np.zeros(matrix.shape)
    matrix_right[:,1:] = matrix[:,:-1]
    matrix_right[:,0] = matrix[:,-1]
    matrix_right = matrix_right[np.newaxis,:]
    #
    displacements = np.concatenate((matrix_down, matrix_up, matrix_left, matrix_right), axis = 0)
    return(displacements)

def calculate_energy(binary_configuration):
    configuration = b2s(binary_configuration)
    displacements = displace_periodic_matrix(configuration)
    energy = np.sum(configuration*np.sum(displacements, axis = 0))
    return(energy)

###############################################################################
#
#                                   Main body
#
###############################################################################
state_size       = 32
max_energy       = (state_size**2)*4
n_bins           = int(max_energy/2)+1

save_directory = "Saved/size"+str(state_size)+"/pickle/"

log_g = np.zeros(n_bins)
H = [np.zeros(n_bins),np.linspace(-max_energy-1,max_energy+1,n_bins+1)]
modification_factor = np.e
final_modification_factor = np.exp(1e-8)
configuration = initialize_spin_configuration()
energy_1 = calculate_energy(configuration)
index_1 = int(np.floor((energy_1+max_energy)/2))
iter_1 = 0
n_rep_state = 0
n_rep_zeros = 0
old_zeros = n_bins
while modification_factor > final_modification_factor:
    iter_2 = 0
    H[0] = np.zeros(n_bins)
    while iter_2 == 0 or not np.all(H[0]/(np.average(H[0])+1.0/n_bins)>0.8):
        if n_rep_state >= 200:
            configuration = initialize_spin_configuration()
            energy_1 = calculate_energy(configuration)
            index_1 = int(np.floor((energy_1+max_energy)/2))
            n_rep_state = 0
        if n_rep_zeros >= 1000:
            configuration = initialize_spin_configuration()
            energy_1 = calculate_energy(configuration)
            index_1 = int(np.floor((energy_1+max_energy)/2))
            n_rep_zeros = 0
        possible_configuration = np.copy(configuration)
        i = np.random.randint(0, state_size)
        j = np.random.randint(0, state_size)
        possible_configuration[i,j] = not possible_configuration[i,j]
        energy_2 = calculate_energy(possible_configuration)
        index_2 = int(np.floor((energy_2+max_energy)/2))
        if index_2 == index_1:
            n_rep_state += 1
        else:
            n_rep_state = 0
        density_ratio = np.exp(log_g[index_1] - log_g[index_2])
        transition_probability = np.min([density_ratio, 1.0])
        r = np.random.rand()
        if r < transition_probability:
            configuration = np.copy(possible_configuration)
            energy_1 = energy_2
            index_1 = index_2
        log_g[index_1] += np.log(modification_factor)
        H[0][index_1] += 1
        iter_2 += 1
        new_zeros = np.sum(H[0]==0)
        if new_zeros == old_zeros:
            n_rep_zeros += 1
        else:
            n_rep_zeros = 0
        if iter_2 % 500000 == 0:
            plt.plot(H[1][:-1]+1,H[0])
            plt.show()
            stdout.write('Accepted Energy = %i, Accepted H = %i, Null Hs = %i \r'\
             % (energy_1,H[0][index_1],new_zeros))
            stdout.flush()
            time.sleep(0.01)
    modification_factor = np.sqrt(modification_factor)
    stdout.write('Iteration = %i, with %i steps\r' % (iter_1, iter_2))
    stdout.flush()
    time.sleep(0.01)

print("")

scale = 2.0/np.exp(g[0])
log_g = log_g + np.log(scale)

# os.mkdir("saved")
pickle.dump(log_g, open(save_directory + "log_density_of_states.p", "wb"))
