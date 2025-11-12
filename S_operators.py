import numpy as np
from scipy.linalg import expm
import functools

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



class S_operators_S_L:
    def __init__(self, Sval, Lval):
        self.Sval = Sval
        self.Lval = Lval
        self.Sdim = int(2*Sval + 1)
        self.Ldim = int(2*Lval + 1)
        self.dim = self.Sdim * self.Ldim
        # define S operators
        self.Sz = np.kron(Szop(Sval), np.eye(self.Ldim))
        self.Sy = np.kron(Syop(Sval), np.eye(self.Ldim))
        self.Sx = np.kron(Sxop(Sval), np.eye(self.Ldim))
        self.Sp = np.kron(Splus(Sval), np.eye(self.Ldim))
        self.Sm = np.kron(Sminus(Sval), np.eye(self.Ldim))
        self.Svec = np.array([self.Sx, self.Sy, self.Sz])
        # define L operators
        self.Lz = np.kron(np.eye(self.Sdim), Szop(Lval))
        self.Ly = np.kron(np.eye(self.Sdim), Syop(Lval))
        self.Lx = np.kron(np.eye(self.Sdim), Sxop(Lval))
        self.Lp = np.kron(np.eye(self.Sdim), Splus(Lval))
        self.Lm = np.kron(np.eye(self.Sdim), Sminus(Lval))
        self.Lvec = np.array([self.Lx, self.Ly, self.Lz])



class MultiSpinOperators:
    """
    Efficient generalized spin operator builder for arbitrary spins.
    Keeps lazy tensor structure to avoid full matrix blow-up until necessary.
    """

    def __init__(self, spins):
        """
        spins: list of tuples [(label, Sval), ...]
        e.g. [('e', 1/2), ('n1', 1/2), ('n2', 1/2)]
        """
        self.spins = spins
        self.labels = [lbl for lbl, _ in spins]
        self.Svals = [S for _, S in spins]
        self.dims = [int(2*S + 1) for S in self.Svals]
        self.dim = np.prod(self.dims)
        self._ops_cache = {}  # store built operators
        if len(spins) >= 2:
            self._assign_SI_aliases()

    def _single_spin_matrix(self, Sval, op):
        """Return single-spin operator matrix."""
        if op == 'Sx': return Sxop(Sval)
        if op == 'Sy': return Syop(Sval)
        if op == 'Sz': return Szop(Sval)
        if op == 'Sp': return Splus(Sval)
        if op == 'Sm': return Sminus(Sval)
        raise ValueError(f"Unknown operator {op}")

    def get_op(self, label, op):
        """Return full Hilbert-space operator for given spin and op (with caching)."""
        key = (label, op)
        if key in self._ops_cache:
            return self._ops_cache[key]

        idx = self.labels.index(label)
        Sval = self.Svals[idx]
        mat = self._single_spin_matrix(Sval, op)
        op_list = [np.eye(d, dtype=complex) for d in self.dims]
        op_list[idx] = mat
        full_op = functools.reduce(np.kron, op_list)
        self._ops_cache[key] = full_op
        return full_op

    def get_Svec(self, label):
        """Return [Sx, Sy, Sz] vector for spin."""
        return np.array([self.get_op(label, 'Sx'),
                         self.get_op(label, 'Sy'),
                         self.get_op(label, 'Sz')])
    
    def get_Sval(self, label=None):
        """
        Retrieve the spin quantum number S for a given label.
        If no label is given, return a dict of all {label: Sval}.
        """
        if label is None:
            return dict(zip(self.labels, self.Svals))
        if label not in self.labels:
            raise ValueError(f"Unknown spin label: {label}")
        return self.Svals[self.labels.index(label)]

    def _assign_SI_aliases(self):
        """Keep old S/I naming for compatibility."""
        e_lbl, n_lbl = self.labels[0], self.labels[1]
        self.Sx = self.get_op(e_lbl, 'Sx')
        self.Sy = self.get_op(e_lbl, 'Sy')
        self.Sz = self.get_op(e_lbl, 'Sz')
        self.Sp = self.get_op(e_lbl, 'Sp')
        self.Sm = self.get_op(e_lbl, 'Sm')
        self.Svec = self.get_Svec(e_lbl)

        self.Ix = self.get_op(n_lbl, 'Sx')
        self.Iy = self.get_op(n_lbl, 'Sy')
        self.Iz = self.get_op(n_lbl, 'Sz')
        self.Ip = self.get_op(n_lbl, 'Sp')
        self.Im = self.get_op(n_lbl, 'Sm')
        self.Ivec = self.get_Svec(n_lbl)



def SO_power(op, power):
    """
    Raises the operator `op` to the specified `power`.
    :param op: Operator matrix (numpy array).
    :param power: Integer power to raise the operator.
    :return: Raised operator matrix.
    """
    if power == 0:
        return np.eye(op.shape[0])  # Identity matrix
    elif power < 0:
        raise ValueError("Negative powers are not supported for operator matrices.")
    else:
        result = np.eye(op.shape[0])  # Start with identity matrix
        for _ in range(power):
            result = result @ op  # Matrix multiplication
        return result
    

def symmprod(opA, opB):
    """
    Computes the symmetric product of two operator matrices.
    :param opA: First operator matrix (numpy array).
    :param opB: Second operator matrix (numpy array).
    :return: Symmetric product matrix.
    """
    return 0.5 * (opA @ opB + opB @ opA)  # Symmetric product


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


############################
# Spin Rotation Operators  #
############################

def R_zyz(so, alpha, beta, gamma):
    """
    Returns the rotation operator for a spin operator in the ZYZ convention. Angles in radians.
    :param so: Spin operator object containing Sx, Sy, Sz matrices.
    :param alpha: Rotation angle around the Z-axis.
    :param beta: Rotation angle around the Y-axis.
    :param gamma: Rotation angle around the new Z-axis.
    :return: Rotation operator matrix.
    """
    return expm(-1j * alpha * so.Sz) @ expm(-1j * beta * so.Sy) @ expm(-1j * gamma * so.Sz)