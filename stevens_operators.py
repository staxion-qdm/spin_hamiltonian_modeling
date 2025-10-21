import numpy as np

from spin_hamiltonian_modeling import S_operators as sops

# constants
h = 4.135667696e-15                 # [eV * s]
muB = 5.788381798170701e-05         # [eV / T]



class StevensOperators:
    def __init__(self, so):        
        # Define constants
        s = so.Sval * (so.Sval + 1)
        cp = 1/2
        cm = 1/(2j)
        II = np.eye(so.dim)

        # Initialize Stevens operators. Where O^{q}_{k} is represented as O_k[q]
        # k = 2
        self.O_2 = {
            -2: cm * (sops.SO_power(so.Sp, 2) - cm * sops.SO_power(so.Sm, 2)),
            -1: cm * sops.symmprod(so.Sz, so.Sp - so.Sm),
            0: 3 * sops.SO_power(so.Sz, 2) - s * II,
            1: cp * sops.symmprod(so.Sz, so.Sp + so.Sm),
            2: cp * (sops.SO_power(so.Sp, 2) + sops.SO_power(so.Sm, 2))
        }

        # k = 4
        self.O_4 = {
            -4: cm * (sops.SO_power(so.Sp, 4) - sops.SO_power(so.Sm, 4)),
            -3: cm * sops.symmprod(so.Sz, sops.SO_power(so.Sp, 3) - sops.SO_power(so.Sm, 3)),
            -2: cm * sops.symmprod(7 * sops.SO_power(so.Sz, 2) - (s + 5)*II, sops.SO_power(so.Sp, 2) - sops.SO_power(so.Sm, 2)),
            -1: cm * sops.symmprod(7 * sops.SO_power(so.Sz, 3) - (3*s + 1)*so.Sz, so.Sp - so.Sm),
            0: 35 * sops.SO_power(so.Sz, 4) - (30*s - 25) * sops.SO_power(so.Sz, 2) + (3*s**2 - 6*s) * II,
            1: cp * sops.symmprod(7 * sops.SO_power(so.Sz, 3) - (3*s + 1)*so.Sz, so.Sp + so.Sm),
            2: cp * sops.symmprod(7 * sops.SO_power(so.Sz, 2) - (s + 5)*II, sops.SO_power(so.Sp, 2) + sops.SO_power(so.Sm, 2)),
            3: cp * sops.symmprod(so.Sz, sops.SO_power(so.Sp, 3) + sops.SO_power(so.Sm, 3)),
            4: cp * (sops.SO_power(so.Sp, 4) + sops.SO_power(so.Sm, 4))
        }

        # k = 6
        self.O_6 = {
            -6: cm * (sops.SO_power(so.Sp, 6) - sops.SO_power(so.Sm, 6)),
            -5: cm * sops.symmprod(so.Sz, sops.SO_power(so.Sp, 5) - sops.SO_power(so.Sm, 5)),
            -4: cm * sops.symmprod(
                11 * sops.SO_power(so.Sz, 2) - (s + 38) * II, 
                sops.SO_power(so.Sp, 4) - sops.SO_power(so.Sm, 4)
                ),
            -3: cm * sops.symmprod(
                11 * sops.SO_power(so.Sz, 3) - (3*s + 59) * so.Sz,
                sops.SO_power(so.Sp, 3) - sops.SO_power(so.Sm, 3)
                ),
            -2: cm * sops.symmprod(
                33 * sops.SO_power(so.Sz, 4)
                - (18*s + 123) * sops.SO_power(so.Sz, 2)
                + (s**2 + 10*s + 102) * II,
                sops.SO_power(so.Sp, 2) - sops.SO_power(so.Sm, 2)
                ),
            -1: cm * sops.symmprod(
                33 * sops.SO_power(so.Sz, 5)
                - (30*s - 15) * sops.SO_power(so.Sz, 3)
                + (5*s**2 - 10*s + 12) * so.Sz,
                so.Sp - so.Sm
                ),
            0: 231 * sops.SO_power(so.Sz, 6)
                - (315*s - 735) * sops.SO_power(so.Sz, 4)
                + (105*s**2 - 525*s + 294) * sops.SO_power(so.Sz, 2)
                - (5*s**3 - 40*s**2 + 60*s) * II,
            1: cp * sops.symmprod(
                33 * sops.SO_power(so.Sz, 5)
                - (30*s - 15) * sops.SO_power(so.Sz, 3)
                + (5*s**2 - 10*s + 12) * so.Sz,
                so.Sp + so.Sm
                ),
            2: cp * sops.symmprod(
                33 * sops.SO_power(so.Sz, 4)
                - (18*s + 123) * sops.SO_power(so.Sz, 2)
                + (s**2 + 10*s + 102) * II,
                sops.SO_power(so.Sp, 2) + sops.SO_power(so.Sm, 2)
                ),
            3: cp * sops.symmprod(
                11 * sops.SO_power(so.Sz, 3) - (3*s + 59) * so.Sz,
                sops.SO_power(so.Sp, 3) + sops.SO_power(so.Sm, 3)
                ),
            4: cp * sops.symmprod(
                11 * sops.SO_power(so.Sz, 2) - (s + 38) * II,
                sops.SO_power(so.Sp, 4) + sops.SO_power(so.Sm, 4)
                ),
            5: cp * sops.symmprod(so.Sz, sops.SO_power(so.Sp, 5) + sops.SO_power(so.Sm, 5)),
            6: cp * (sops.SO_power(so.Sp, 6) + sops.SO_power(so.Sm, 6))
        }