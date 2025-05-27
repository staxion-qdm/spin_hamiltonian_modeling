import numpy as np
import itertools as itt
from matplotlib.collections import LineCollection
from spectroscopy_analysis_functions import S_operators as sops
import scipy as scp



#########################################################
### DEFINE MATRICES AND FUNCTIONS FOR THE HAMILTONIAN ###
#########################################################

# define the scaler parameters
h = 4.135667696e-15                 # [eV * s]
gxx = 1.96                          # [1]
gyy = 1.96                          # [1]
gzz = 1.9619                        # [1]
g = 1.961                           # [1]
ge = 2.00231930436182               # [1]
mu_B = 5.788381798170701e-05        # [eV / T]
b20 = 1.75e3 / 10e3 * g * mu_B      # [eV]
b22 = -0.38e3 / 10e3 * g * mu_B     # [eV]
A51 = 92 / 10e3 * g * mu_B          # [eV]
A19x = 0                            # [eV]
A19y = 0                            # [eV]
A19z = 6 / 10e3 * g * mu_B          # [eV]



# defint the 3D rotation matrices
def Rz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0,0,1]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0,1,0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rx(theta):
    return np.array([[1,0,0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def Rotmat(alpha, beta, theta):
    return Rz(theta) @ Ry(beta) @ Rx(alpha)



# define the D matrix
def Dmat(alpha, beta, theta):
    D = b20/h                                                               # [Hz]   
    E = b22/h                                                               # [Hz]
    Dur = np.array([[-D/3+E,0,0],
                    [0,-D/3-E,0],
                    [0,0,2*D/3]])                                           # [Hz]
    Drot = Rotmat(alpha, beta, theta) @ Dur @ Rotmat(alpha, beta, theta).T  # [Hz]
    return Drot

#define the g matrix
def gmat(alpha, beta, theta):
    gmat0 = np.array([[gxx, 0, 0],
                     [0, gyy, 0],
                     [0, 0, gzz]])
    return Rotmat(alpha, beta, theta) @ gmat0 @ Rotmat(alpha, beta, theta).T

# define total Hamiltonian
def totalHam3o2c7o2_rotation(Bzval, alpha, beta, theta):
    # magnetic field vector
    Bvec = np.array([0,0,Bzval])
    # zeeman Hamiltonian
    Hzee = beta*np.einsum('i,ij,jkl->kl', Bvec, gmat(alpha, beta, theta), sops.Svec(1, 3/2, 8))
    # zero-field splitting Hamiltonian
    SdotDdotS = np.einsum('ijk,il,lkm->jm', sops.Svec(1, 3/2, 8), Dmat(alpha, beta, theta), sops.Svec(1, 3/2, 8))
    Hzf = h * SdotDdotS
    # hyperfine interaction Hamiltonian
    Hhp = np.zeros((32,32))
    for i in range(3):
        Hhp = Hhp + A51*np.kron(sops.Svec(1, 3/2, 1)[i], sops.Svec(1, 7/2, 1)[i])

    totalHam = Hzee + Hzf + Hhp
    
    return totalHam



# define total Hamiltonian for single value solving
def totalHam3o2c7o2_rotation_A(Bzval, alpha, beta, theta, A51):
    # magnetic field vector
    Bvec = np.array([0,0,Bzval])
    # zeeman Hamiltonian
    Hzee = mu_B * np.einsum('i,ij,jkl->kl', Bvec, gmat(alpha, beta, theta), sops.Svec(1, 3/2, 8))
    # Hzee = beta*g*Bzval*np.kron(sops.Szop(3/2), np.eye(8))
    # zero-field splitting Hamiltonian
    SdotDdotS = np.einsum('ijk,il,lkm->jm', sops.Svec(1, 3/2, 8), Dmat(alpha, beta, theta), sops.Svec(1, 3/2, 8))
    Hzf = h * SdotDdotS
    # hyperfine interaction Hamiltonian
    Hhp = np.zeros((32,32))
    for i in range(3):
        Hhp = Hhp + A51*np.kron(sops.Svec(1, 3/2, 1)[i], sops.Svec(1, 7/2, 1)[i])
    Hhp = h * Hhp
    # # superhyperfine interaction Hamiltonian
    # Hshp = np.zeros((32,32))
    # A19 = np.array([A19x, A19y, A19z])
    # for i in range(3):
    #     Hshp = Hshp + A19[i]*np.kron(sops.Svec(1, 3/2, 1)[i], sops.Svec(1, 7/2, 1)[i])

    totalHam = Hzee + Hzf + Hhp #+ Hshp
    
    return totalHam



# define the function to calculate the term proportional to probability of transition between two states
def mwp_spin_int_prob(state_i, state_f, alpha, beta, theta):
    gdots = np.einsum('ij,jkl->ikl', gmat(alpha, beta, theta), sops.Svec(1, 3/2, 8))[0]
    prob = np.abs(np.vdot(state_f, gdots @ state_i))**2
    return prob


##########################################
# DEFINT THE ENERGY TRANSITION FUNCTIONS #
##########################################


# get all possible energy transitions from eigenvalues and eigenvectors lists (faster function)
def possible_energy_transitions(eigvals, eigvecs, hamdim, alpha, beta, theta):
    # get all index combinations
    eigvinds_combinations = itt.combinations(np.linspace(0,hamdim - 1,hamdim).astype(int), 2)

    # to append the array [eigenvalue difference, transition probability, index pair]
    transition_info = []

    for combs in eigvinds_combinations:
        # find the probability of transition
        # transit_prob = np.abs(np.vdot(eigvecs[:,combs[0]], Hmw @ eigvecs[:,combs[1]]))**2
        # sxprob = np.abs(np.vdot(eigvecs[:,combs[0]], np.kron(Sxop(3/2), np.eye(8)) @ eigvecs[:,combs[1]]))**2
        # syprob = np.abs(np.vdot(eigvecs[:,combs[0]], np.kron(Syop(3/2), np.eye(8)) @ eigvecs[:,combs[1]]))**2
        splusprob = np.abs(np.vdot(eigvecs[:,combs[0]], np.kron(sops.Splus(3/2), np.eye(8)) @ eigvecs[:,combs[1]]))**2
        sminusprob = np.abs(np.vdot(eigvecs[:,combs[0]], np.kron(sops.Sminus(3/2), np.eye(8)) @ eigvecs[:,combs[1]]))**2
        
        # transit_prob = sxprob + syprob
        transit_prob = splusprob + sminusprob
        # transit_prob = mwp_spin_int_prob(eigvecs[:,combs[0]], eigvecs[:,combs[1]], alpha, beta, theta)

        # append the information
        energy_diff = np.abs(eigvals[combs[0]] - eigvals[combs[1]])
        transition_info.append(np.array([energy_diff, transit_prob, combs[0], combs[1]]))

    return np.array(transition_info)



# solve Hamiltonian for an array of parameter values
def solveHam_3o2_7o2_rotations(B, alpha, beta, theta):
    # initialize the eigenvalue and eigenvector arrays
    hamdim = np.shape(totalHam3o2c7o2_rotation(1, np.pi, np.pi, np.pi))[0]   # dimensions of the Hamiltonian
    Edim = int(scp.special.comb(hamdim, 2))
    eigval = np.zeros((len(B), len(alpha), len(beta), len(theta), hamdim)).astype(np.complex128)          
    eigvec = np.zeros((len(B), len(alpha), len(beta), len(theta), hamdim, hamdim)).astype(np.complex128)
    Earray = np.zeros((len(B), len(alpha), len(beta), len(theta), Edim))
    eprobarray = np.zeros((len(B), len(alpha), len(beta), len(theta), Edim))
    epairinds = np.zeros((len(B), len(alpha), len(beta), len(theta), Edim, 2))

    # solve for the eigenvalues and the eigenvectors
    for bind in range(len(B)):
        for alphind in range(len(alpha)):
            for betind in range(len(beta)):
                for theind in range(len(theta)):
                    Htotal = totalHam3o2c7o2_rotation(B[bind], alpha[alphind], beta[betind], theta[theind])
                    eigenvalue, eigenvector = np.linalg.eig(Htotal)
                    transition_info = possible_energy_transitions(eigenvalue, eigenvector, hamdim, alpha[alphind], beta[betind], theta[theind])
                    eigval[bind, alphind, betind, theind, :] = eigenvalue
                    eigvec[bind, alphind, betind, theind, :, :] = eigenvector
                    Earray[bind, alphind, betind, theind, :] = transition_info[:,0]
                    eprobarray[bind, alphind, betind, theind, :] = transition_info[:,1]
                    epairinds[bind, alphind, betind, theind, :, :] = transition_info[:,2:]
                        
    return eigval, eigvec, Earray, eprobarray, epairinds



# solve Hamiltonian for single parameter values and a range of B values
def solveHam_3o2_7o2_rotations_sv(B, alpha, beta, theta, A51):
    # initialize the eigenvalue and eigenvector arrays
    hamdim = np.shape(totalHam3o2c7o2_rotation(1, np.pi, np.pi, np.pi))[0]   # dimensions of the Hamiltonian
    Edim = int(scp.special.comb(hamdim, 2))
    eigval = np.zeros((len(B), hamdim)).astype(np.complex128)          
    eigvec = np.zeros((len(B), hamdim, hamdim)).astype(np.complex128)
    Earray = np.zeros((len(B), Edim))
    eprobarray = np.zeros((len(B), Edim))
    epairinds = np.zeros((len(B), Edim, 2))

    # solve for the eigenvalues and the eigenvectors
    for bind in range(len(B)):
        Htotal = totalHam3o2c7o2_rotation_A(B[bind], alpha, beta, theta, A51)
        eigenvalue, eigenvector = np.linalg.eig(Htotal)
        transition_info = possible_energy_transitions(eigenvalue, eigenvector, hamdim, alpha, beta, theta)
        eigval[bind, :] = eigenvalue
        eigvec[bind, :, :] = eigenvector
        Earray[bind, :] = transition_info[:,0]
        eprobarray[bind, :] = transition_info[:,1]
        epairinds[bind, :, :] = transition_info[:,2:]
                        
    return eigval, eigvec, Earray, eprobarray, epairinds



# get the spin states with non-zero coefficients of a vector
def get_spin_states_3o2_7o2(state_vector):
    # non-zero coefficients
    nzcoeff = (np.argwhere(np.abs(state_vector) > 1e-15)).flatten()
    # non-zero I index
    nzindI = (np.argwhere(np.abs(state_vector) > 1e-15) % 8).flatten()
    # non-zero S index
    nzindS = (np.argwhere(np.abs(state_vector) > 1e-15) // 8).flatten()
    # 2*I value
    I2 = 7 - 2*nzindI
    # 2*S value
    S2 = 3 - 2*nzindS
    # format as string
    states = ''
    for i in range(len(I2)):
        state = str(np.round(state_vector[nzcoeff][i],4)) + '|'+str(S2[i])+'/2,'+str(I2[i])+'/2>+'
        states += state
    return states



# sort energy transition points into arrays inside a bound
def sortenergies(energydata, probability_data, index_data, lower_bound, upper_bound):
    # a list to append transition values to 
    doi_e = []
    # a list to append length of each
    doi_e_len = []
    # a list to append the reordered indices
    reord_inds = []


    # loop through initial index to final index
    for i in range(np.shape(energydata)[0]):
        # non-zero probability transitions
        nztr = np.transpose(np.argwhere((probability_data[i] > lower_bound) & (probability_data[i] < upper_bound)))[0]
        # sort the data by energy level
        sedata = np.argsort(energydata[i][nztr])
        # append transitions
        doi_e.append(energydata[i][nztr][sedata])
        doi_e_len.append(len(energydata[i][nztr][sedata]))
        # append the indices
        reord_inds.append(index_data[i][nztr][sedata])

    doi_e_len = np.array(doi_e_len)

    # prepare a list of lists to append plotting points
    plotlines = [[] for _ in range(np.min(doi_e_len))]
    # prepare a list of eigenvector index lists
    eigvind = [[] for _ in range(np.min(doi_e_len))]

    for j in range(np.min(doi_e_len)):
        for n in range(np.shape(energydata)[0]):
            plotlines[j].append(doi_e[n][j])
            eigvind[j].append(reord_inds[n][j])

    return plotlines, eigvind




# plot the ordered energy transitions
def transparent_line_plot(ax, x, ylist, opaqueness_list, xylimits=None):

    for i in range(ylist.shape[1]):  # Iterate over columns of ylist
        y = ylist[:, i]
        opaqueness = opaqueness_list[:, i]

        # Prepare line segments
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a custom colormap for transparency
        colors = plt.cm.Greys(np.ones(len(opaqueness)))
        colors[:, -1] = opaqueness  # Modify alpha (transparency)

        # Create LineCollection
        lc = LineCollection(segments, colors=colors, linestyle='--')

        # Add LineCollection to the current axes
        ax.gca().add_collection(lc)

    # Set x and y limits if specified
    if xylimits is not None:
        ax.set_xlim(xylimits[0, 0], xylimits[0, 1])
        ax.set_ylim(xylimits[1, 0], xylimits[1, 1])