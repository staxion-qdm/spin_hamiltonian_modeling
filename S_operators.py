import numpy as np

def Splus(S):
    # initialize matrix and m values
    numm = int(2*S + 1)
    Splusmat = np.zeros((numm, numm))
    mvals = np.linspace(-S, S, numm)

    # fill in the values
    for i in range(numm):
        for j in range(numm):
            Splusmat[i, j] = np.identity(numm + 1)[i+1, j] * np.sqrt(S*(S+1) - mvals[i]*mvals[j])

    return Splusmat


def Sminus(S):
    # initialize matrix and m values
    numm = int(2*S + 1)
    Splusmat = np.zeros((numm, numm))
    mvals = np.linspace(-S, S, numm)

    # fill in the values
    for i in range(numm):
        for j in range(numm):
            Splusmat[i, j] = np.identity(numm + 1)[i, j+1] * np.sqrt(S*(S+1) - mvals[i]*mvals[j])

    return Splusmat


def Sxop(S):
    Sxop = 1/2*(Splus(S) + Sminus(S))
    return Sxop


def Syop(S):
    Syop = 1/(1j*2)*(Splus(S) - Sminus(S))
    return Syop


def Szop(S):
    # initialize m values
    numm = int(2*S + 1)
    Szop = np.zeros((numm, numm))
    mvals = np.linspace(S, -S, numm)

    # fill in matrix values
    for i in range(numm):
        Szop[i,i] = mvals[i]

    return Szop


def Svec(leye, S, reye):
    Sxext = np.kron(np.kron(np.eye(leye), Sxop(S)), np.eye(reye))
    Syext = np.kron(np.kron(np.eye(leye), Syop(S)), np.eye(reye))
    Szext = np.kron(np.kron(np.eye(leye), Szop(S)), np.eye(reye))
    return np.array([Sxext, Syext, Szext])



class S_operators_S_I:
    def __init__(self, Sval, Ival):
        self.Sval = Sval
        self.Ival = Ival
        self.Sdim = int(2*Sval + 1)
        self.Idim = int(2*Ival + 1)
        self.dim = self.Sdim * self.Idim
        # define S operators
        self.Sz = np.kron(Szop(Sval), np.eye(self.Idim))
        self.Sy = np.kron(Syop(Sval), np.eye(self.Idim))
        self.Sx = np.kron(Sxop(Sval), np.eye(self.Idim))
        self.Sp = np.kron(Splus(Sval), np.eye(self.Idim))
        self.Sm = np.kron(Sminus(Sval), np.eye(self.Idim))
        self.Svec = np.array([self.Sx, self.Sy, self.Sz])
        # define I operators
        self.Iz = np.kron(np.eye(self.Sdim), Szop(Ival))
        self.Iy = np.kron(np.eye(self.Sdim), Syop(Ival))
        self.Ix = np.kron(np.eye(self.Sdim), Sxop(Ival))
        self.Ip = np.kron(np.eye(self.Sdim), Splus(Ival))
        self.Im = np.kron(np.eye(self.Sdim), Sminus(Ival))
        self.Ivec = np.array([self.Ix, self.Iy, self.Iz])

#############################
# Rotation Matrix Operators #
#############################

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