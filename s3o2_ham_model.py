import numpy as np
import itertools as itt
from spectroscopy_analysis_functions import S_operators as sops



# get all possible energy transitions from eigenvalues and eigenvectors lists (faster function)
def energy_transitions_np(eigvecs, eigvals, Sval):
    # get all index combinations
    hamdim = np.shape(eigvecs)[0]
    eigvinds_combinations = itt.combinations(np.linspace(0,hamdim - 1,hamdim).astype(int), 2)

    # to append the array [eigenvalue difference, transition probability, index pair]
    transition_info = []

    for combs in eigvinds_combinations:
        # find the probability of transition
        splusprob = np.abs(np.vdot(eigvecs[:,combs[0]], sops.Splus(Sval) @ eigvecs[:,combs[1]]))**2
        sminusprob = np.abs(np.vdot(eigvecs[:,combs[0]], sops.Sminus(Sval) @ eigvecs[:,combs[1]]))**2        
        transit_prob = splusprob + sminusprob

        # append the information
        energy_diff = np.abs(eigvals[combs[0]] - eigvals[combs[1]])
        transition_info.append(np.array([energy_diff, transit_prob, combs[0], combs[1]]))

    return np.array(transition_info)



# define the scaler parameters
h = 4.135667696e-15                 # [eV * s]
muB = 5.7883818060e-5               # [eV / T]
Sval = 3/2
Ide = np.eye(int(2*Sval + 1))

def LIPHam(Bz, ge, DD, EE):
    HamZ = muB*ge*Bz*sops.Szop(Sval)
    HamZF = h * DD * (sops.Szop(Sval) @ sops.Szop(Sval) - Sval*(Sval + 1)/3 * Ide) + h * EE * (sops.Sxop(Sval) @ sops.Sxop(Sval) - sops.Syop(Sval) @ sops.Syop(Sval))
    THam = HamZ + HamZF
    return THam

def LIPspectra(Bvals, ge, DD, EE):
    # initialize arrays to append for each B point
    e_transitions = []
    e_prob = []
    transition_inds = []
    
    # calculate eigenenergies for each B and reorder according to energy values
    for i in range(len(Bvals)):
        # calculate eigenstates and eigenvalues of the hamiltonian
        Ham = LIPHam(Bvals[i], ge, DD, EE)
        eigvals, eigvecs = np.linalg.eigh(Ham)
        
        # append transition information
        t_info = energy_transitions_np(eigvecs, eigvals, Sval)
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