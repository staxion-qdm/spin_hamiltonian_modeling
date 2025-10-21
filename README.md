# Spin Hamiltonian Modeling in Python
Modeling the interactions between spins and magnetic fields to predict ESR spectroscopy data.

# Examples
## Hamiltonians With Zeeman Operators
[This](https://iopscience.iop.org/article/10.1143/JJAP.10.1497/pdf) paper uses the following spin Hamiltonian:

$$ \hat{H} = g \mu_{B} \mathbf{B} \cdot \mathbf{\hat{S}} + \frac{a}{6} \left[ \hat{S}_{x}^{4} + \hat{S}_{y}^{4} + \hat{S}_{z}^{4} - \frac{1}{5} S (S + 1) (3S^{2} + 3S - 1) \right] $$

Where $S = 5/2$ and $S_{z}$ is along the [001] direction.

First, define Hamilonian class in terms of the electron and nuclear spin operators (defined in S_operators.py):

```
from spin_hamiltonian_modeling import S_operators as sop
from spin_hamiltonian_modeling import esr_spectra as esrsp

# constants
h = 4.135667696e-15                     # [eV * s]
muB = 5.788381798170701e-05             # [eV / T]

class Fe3plHam:
    def __init__(self, so, g, a):
        self.so = so
        self.g = g
        self.a = a                      # [Hz]
        self.gmat = g * np.eye((3))

    def Ham(self, Bz):
        # Define the B field vector in T
        self.Bvec = np.array([0, 0, Bz])
        # Define the Hamiltonians in eV
        self.ham_zeeman = muB * np.einsum('i,ij,jkl->kl', self.Bvec, self.gmat, self.so.Svec)
        self.ham_zfs = h * self.a/6 * (sop.SO_power(self.so.Sx, 4) + sop.SO_power(self.so.Sy, 4) + sop.SO_power(self.so.Sz, 4) - 1/5 * self.so.Sval*(self.so.Sval + 1)*(3*self.so.Sval**2 + 3*self.so.Sval - 1))
        self.total_ham = self.ham_zeeman + self.ham_zfs
        return self.total_ham
```

Then compute all possible transition lines with their corresponding probabilities:

```
# Spin operators initialisation
so = sop.S_operators_S_I(5/2, 0)

# B values
Bzvals = np.linspace(0, 0.1, 101)

# Define Hamiltonian and get transition frequencies
oham = Fe3plHam(so, g, a)
ftran, eprob = esrst.esr_spectra(oham, Bzvals, temperature=temp)
ftran = ftran / 1e9  # Convert to GHz
```

Where ftran (transition frequency array) and eprob (corresponding probability proportional values) are shape (len(Bzvals), len(transitions))

## Hamiltonians With Stevens Operators

Hamiltonian used for $\text{Eu}^{2+}$ in [this](https://www.jstor.org/stable/pdf/100549.pdf?utm_source=chatgpt.com) paper:

$$ \hat{H} = g \mu_{B} \mathbf{B} \cdot \mathbf{\hat{S}} + B_{4}(\hat{O}_{4}^{0} + 5 \hat{O}_{4}^{4}) + B_{6}(\hat{O}_{6}^{0} - 21 \hat{O}_{6}^{4}) + A \mathbf{\hat{S}} \cdot \mathbf{\hat{I}} $$

Where the effective spin $S=\frac{7}{2}$ and $I=\frac{5}{2}$ is used

First, define Hamilonian class in terms of the electron and nuclear spin operators (defined in S_operators.py) and Stevens operators (defined in stevens_operators.py):

```
from spin_hamiltonian_modeling import S_operators as sop
from spin_hamiltonian_modeling import stevens_operators as stop
from spin_hamiltonian_modeling import esr_spectra as esrsp

# Constants
h = 4.135667696e-15                 # [eV * s]
muB = 5.788381798170701e-05         # [eV / T]
wltohz = 29.9792458e9               # [cm * Hz]         

class Eu2pHam:
    def __init__(self, so, sto, g, B4, B6, A):
        self.so = so
        self.sto = sto
        self.g = g
        # Convert coefficients from [10^-4 cm^-1] to [Hz]
        self.B4 = B4 * 1e-4 * wltohz          # [Hz]
        self.B6 = B6 * 1e-4 * wltohz          # [Hz]
        self.A = A * 1e-4 * wltohz            # [Hz]

    def Ham(self, Bz):
        # Define the Hamiltonians in eV
        self.ham_zeeman = muB * self.g * Bz * self.so.Sz
        self.ham_zfs = h * self.B4 * (self.sto.O_4[0] + 5*self.sto.O_4[4]) + h * self.B6 * (self.sto.O_6[0] - 21*self.sto.O_6[4])
        self.ham_hf = h * self.A * sum(S @ I for S, I in zip(self.so.Svec, self.so.Ivec))
        self.total_ham = self.ham_zeeman + self.ham_zfs + self.ham_hf
        return self.total_ham
```

Then compute all possible transition lines with their corresponding probabilities:

```
# Spin operators initialisation
so = sop.S_operators_S_I(7/2, 5/2)
sto = stop.StevensOperators(so)

# B values
Bzvals = np.linspace(0, 1, 101)

# Define Hamiltonian and get transition frequencies
oham = Eu2pHam(so, sto, g, B4, B6, A)
ftran, eprob = esrst.esr_spectra(oham, Bzvals, temperature=temp)
ftran = ftran / 1e9  # Convert to GHz
```

Where ftran (transition frequency array) and eprob (corresponding probability proportional values) are shape (len(Bzvals), len(transitions))

### Coefficient Adjustable Plotting Example with ipympl
Install ipympl to plot and adjust coefficients in real time. Example:

```
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipw
from functools import lru_cache

so = sop.S_operators_S_I(7/2, 5/2)
sto = stop.StevensOperators(so)
Bzvals = np.linspace(0, 1.5, 151)

# --- caching wrapper for the expensive computation ---
# round parameters so cache keys are limited in size; adjust decimals as needed
@lru_cache(maxsize=128)
def cached_esr(g_r, B4_r, B6_r, A_r, temp_r):
    # rebuild ham and compute spectra
    oham = Eu2pHam(so, sto, g_r, B4_r, B6_r, A_r)
    ftran, eprob = esrst.esr_spectra(oham, Bzvals, temperature=temp_r)
    return ftran / 1e9, eprob  # convert to GHz here

# --- Build figure and plot static experimental data ONCE ---
%matplotlib widget
fig, ax = plt.subplots(figsize=(12, 6))

ax.set_xlim(0, Bzvals[-1])
ax.set_ylim(0, 20)
ax.set_xlabel('B (T)')
ax.set_ylabel('Frequency (GHz)')
ax.legend()

# Create placeholder line artists for simulation: determine n_trans from an initial evaluation
# initial params (your defaults)
g0 = 1.96
g0mk = 2.13
B40 = 55.75
B60 = 0.25
A0 = 34.07
temp0 = 10

# Get initial spectrum (rounded values for cache key)
decimals_cache = 4
f0, e0 = cached_esr(
    round(g0, decimals_cache),
    round(B40, decimals_cache),
    round(B60, decimals_cache),
    round(A0, decimals_cache),
    round(temp0, decimals_cache)
)

ntrans = f0.shape[1]

# create Line2D artists once (lightweight)
lines = []
for i in range(ntrans):
    (line,) = ax.plot(Bzvals, f0[:, i], lw=1.2, alpha=1.0, color='black')
    lines.append(line)

fig.canvas.draw()

# --- update function that only updates the line data (fast) ---
def update_sim(g, B4, B6, A, temp, cscale, interp_points=30):
    # round for cache key (coarser rounding gives more cache hits)
    g_r = round(g, decimals_cache)
    B4_r = round(B4, decimals_cache)
    B6_r = round(B6, decimals_cache)
    A_r = round(A, decimals_cache)
    temp_r = round(temp, decimals_cache)

    ftran, eprob = cached_esr(g_r, B4_r, B6_r, A_r, temp_r)

    # If number of transitions changed (unlikely), adjust lines list:
    if ftran.shape[1] != len(lines):
        # quick fallback: clear and recreate (rare)
        for ln in lines:
            ln.remove()
        lines.clear()
        for i in range(ftran.shape[1]):
            (line,) = ax.plot(Bzvals, ftran[:, i], lw=1.2, alpha=1.0, color='black')
            lines.append(line)
    else:
        # update existing artists' data
        for i, ln in enumerate(lines):
            ln.set_xdata(Bzvals)
            ln.set_ydata(ftran[:, i])
            # set alpha from the average transition probability scaled by cscale
            # (per-point alpha is heavier; approximate with mean)
            avgprob = np.clip(np.mean(eprob[:, i]), 0.0, 1.0)
            ln.set_alpha(avgprob * cscale)

    # optional: update legend text / annotation if you want
    fig.canvas.draw_idle()

# --- build sliders with continuous_update=False to avoid updates while dragging ---
layout = ipw.Layout(width='1000px')
sliders = {
    'g': ipw.FloatSlider(min=1.0, max=3.0, step=0.001, value=g0mk, description='g', layout=layout, continuous_update=False),
    'B4': ipw.FloatSlider(min=54, max=60, step=0.001, value=B40, description='B4 (10^-4 cm^-1)', layout=layout, continuous_update=False),
    'B6': ipw.FloatSlider(min=0.2, max=0.6, step=0.001, value=B60, description='B6 (10^-4 cm^-1)', layout=layout, continuous_update=False),
    'A': ipw.FloatSlider(min=15, max=35, step=0.01, value=A0, description='A (10^-4 cm^-1)', layout=layout, continuous_update=False),
    'temp': ipw.FloatSlider(min=0, max=300, step=0.01, value=temp0, description='Temp (K)', layout=layout, continuous_update=False),
    'cscale': ipw.FloatSlider(min=0, max=1, step=0.01, value=1, description='Color scale', layout=layout, continuous_update=False),
}

out = ipw.interactive_output(update_sim, sliders)

# display sliders and connect
controls = ipw.VBox(list(sliders.values()))
display(controls, out)

```
