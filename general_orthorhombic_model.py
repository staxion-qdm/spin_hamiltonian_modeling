import numpy as np

from spin_hamiltonian_modeling import S_operators as sops
from spin_hamiltonian_modeling import stevens_operators as stops

# constants
h = 4.135667696e-15                 # [eV * s]
muB = 5.788381798170701e-05         # [eV / T]

class orthorhom_ham:
    def __init__(self, so, g, E, D, A):
        self.so = so
        self.g = g
        self.E = E
        self.D = D
        self.A = A

    def Ham(self, Bz):
        # define the hamiltonians
        ham_zeeman = muB * self.g * Bz * self.so.Sz
        ham_zf = h * self.D * (self.so.Sz @ self.so.Sz - (self.so.Sval*(self.so.Sval + 1)/3)*np.eye(self.so.dim)) + h * self.E * (self.so.Sx @ self.so.Sx - self.so.Sy @ self.so.Sy)
        ham_hf = h * self.A * (self.so.Sx @ self.so.Ix + self.so.Sy @ self.so.Iy + self.so.Sz @ self.so.Iz)
        total_ham = ham_zeeman + ham_zf + ham_hf
        return total_ham
    


class orthorhom_ham_rot:
    def __init__(self, so, gmat, Dmat, A, alpha, beta, theta):
        # convert from degrees to radians
        alpha_rad = np.deg2rad(alpha)
        beta_rad = np.deg2rad(beta)
        theta_rad = np.deg2rad(theta)
        # initialise the constants
        self.so = so
        self.gmat = gmat
        self.Dmat = Dmat
        self.A = A
        self.Bvec = np.array([0, 0, 0])
        self.Rmat = sops.Rotmat(alpha_rad, beta_rad, theta_rad)
        # define rotated matrices
        self.gmat_rot = self.Rmat @ self.gmat @ self.Rmat.T
        self.Dmat_rot = self.Rmat @ self.Dmat @ self.Rmat.T
        # initialize hamiltonians
        self.ham_zeeman = 0
        self.ham_zf = 0
        self.ham_hf = 0

    def Ham(self, Bz):
        # define the B field vector
        self.Bvec = np.array([0, 0, Bz])
        # define the hamiltonians
        self.ham_zeeman = muB * np.einsum('i,ij,jkl->kl', self.Bvec, self.gmat_rot, self.so.Svec)
        self.ham_zf = h * np.einsum('ikl,ij,jlm->km', self.so.Svec, self.Dmat_rot, self.so.Svec)
        self.ham_hf = h* self.A * (self.so.Sx @ self.so.Ix + self.so.Sy @ self.so.Iy + self.so.Sz @ self.so.Iz)
        total_ham = self.ham_zeeman + self.ham_zf + self.ham_hf
        return total_ham
    


class stevens_op_ham:
    def __init__(self, so, sto, k2coefs, k4coefs, k6coefs, gmat):
        self.so = so
        self.sto = sto
        self.k2coefs = k2coefs
        self.k4coefs = k4coefs
        self.k6coefs = k6coefs
        self.gmat = gmat

    def Ham(self, Bz):
        # Define the B field vector
        self.Bvec = np.array([0, 0, Bz])                                                        # [T]
        # Define the Hamiltonians (energy units)
        self.ham_zeeman = muB * np.einsum('i,ij,jkl->kl', self.Bvec, self.gmat, self.so.Svec)   # [eV]   
        # Define the ZFS Hamiltonian using Stevens operators
        qarray = np.linspace(-6, 6, 13).astype(int)
        self.ham_zfs = 0
         # Loop through the Stevens operators
        for i, val in enumerate(qarray):
            i4 = 0
            i2 = 0
            self.ham_zfs += self.k6coefs[i] * self.sto.O_6[val] 
            if (-4 <= val <= 4):
                self.ham_zfs += self.k4coefs[i4] * self.sto.O_4[val]
                i4 += 1
                if (-2 <= val <= 2):
                    self.ham_zfs += self.k2coefs[i2] * self.sto.O_2[val]
                    i2 += 1
        total_ham = self.ham_zeeman + self.ham_zfs
        return total_ham
