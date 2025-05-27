import numpy as np
import itertools as itt
from matplotlib.collections import LineCollection
from spectroscopy_analysis_functions import S_operators as sops
import scipy as scp


# constants
h = 4.135667696e-15                 # [eV * s]
muB = 5.7883818060e-5               # [eV / T]


# get all possible energy transitions from eigenvalues and eigenvectors lists (faster function)
def energy_transitions_np(Ham, eigvecs, eigvals):
    # get all index combinations
    hamdim = Ham.so.dim
    eigvinds_combinations = itt.combinations(np.linspace(0,hamdim - 1,hamdim).astype(int), 2)

    # to append the array [eigenvalue difference, transition probability, index pair]
    transition_info = []

    for combs in eigvinds_combinations:
        # find the probability of transition
        sxprob = np.abs(np.vdot(eigvecs[:,combs[0]], Ham.so.Sp @ eigvecs[:,combs[1]]))**2
        syprob = np.abs(np.vdot(eigvecs[:,combs[0]], Ham.so.Sm @ eigvecs[:,combs[1]]))**2        
        transit_prob = sxprob + syprob

        # append the information
        energy_diff = np.abs(eigvals[combs[0]] - eigvals[combs[1]])
        transition_info.append(np.array([energy_diff, transit_prob, combs[0], combs[1]]))

    return np.array(transition_info)


def esr_spectra(Ham_proc, Bz_vals):
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
        t_info = energy_transitions_np(Ham_proc, eigvecs, eigvals)
        # get sorted index for energy transitions
        sortEind = np.argsort(t_info[:,0])
        # sort the energy transitions
        e_transitions.append(t_info[sortEind,0])
        # sort the transition probabilities using the sorted index for energy transitions
        e_prob.append(t_info[sortEind,1])
        transition_inds.append([t_info[sortEind,2:3], t_info[sortEind,2:3]])

    # convert energy to frequency in GHz
    ftran = np.array(e_transitions)/(h*1e9)
    # normalize the probability
    eprob = np.array(e_prob)/np.max(np.array(e_prob))

    return ftran, eprob