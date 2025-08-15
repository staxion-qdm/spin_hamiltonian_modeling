import numpy as np
import itertools as itt
from matplotlib.collections import LineCollection
from spin_hamiltonian_modeling import S_operators as sops
import scipy as scp


# constants
h = 4.135667696e-15                 # [eV * s]
muB = 5.7883818060e-5               # [eV / T]
kB = 8.617333262e-5                 # [eV / K]


# get all possible energy transitions from eigenvalues and eigenvectors lists (faster function)
def energy_transitions_np(Ham, eigvecs, eigvals, temperature):
    # get all index combinations
    hamdim = Ham.so.dim
    eigvinds_combinations = itt.combinations(np.linspace(0,hamdim - 1,hamdim).astype(int), 2)

    # compute Boltzmann factors
    boltzmann_factors = np.exp(-eigvals / (kB * temperature))
    partition_func = np.sum(boltzmann_factors)

    # to append the array [eigenvalue difference, transition probability, index pair]
    transition_info = []

    for i, j in eigvinds_combinations:
        # Transition probability
        sxprob = np.abs(np.vdot(eigvecs[:, i], Ham.so.Sp @ eigvecs[:, j]))**2
        syprob = np.abs(np.vdot(eigvecs[:, i], Ham.so.Sm @ eigvecs[:, j]))**2
        transit_prob = sxprob + syprob

        # Boltzmann-weighted intensity (from lower-energy state)
        if eigvals[i] < eigvals[j]:
            weight = boltzmann_factors[i] / partition_func
        else:
            weight = boltzmann_factors[j] / partition_func

        weighted_prob = transit_prob * weight

        energy_diff = np.abs(eigvals[i] - eigvals[j])
        transition_info.append(np.array([energy_diff, weighted_prob, i, j]))

    return np.array(transition_info)


def esr_spectra(Ham_proc, Bz_vals, temperature=0.01):
    # initialize arrays to append for each B point
    e_transitions = []
    e_prob = []
    transition_inds = []
    
    # calculate eigenenergies for each B and reorder according to energy values
    for i in range(len(Bz_vals)):
        # calculate eigenstates and eigenvalues of the hamiltonian
        Ham = Ham_proc.Ham(Bz_vals[i])
        eigvals, eigvecs = np.linalg.eigh(Ham)
        
        # append transition information
        t_info = energy_transitions_np(Ham_proc, eigvecs, eigvals, temperature=temperature)
        # get sorted index for energy transitions
        sortEind = np.argsort(t_info[:,0])
        # sort the energy transitions
        e_transitions.append(t_info[sortEind,0])
        # sort the transition probabilities using the sorted index for energy transitions
        e_prob.append(t_info[sortEind,1]/np.max(t_info[sortEind,1]))
        transition_inds.append([t_info[sortEind,2:3], t_info[sortEind,2:3]])

    # convert energy to frequency in GHz
    ftran = np.array(e_transitions)/(h*1e9)
    # normalize the probability
    eprob = np.array(e_prob)

    return ftran, eprob