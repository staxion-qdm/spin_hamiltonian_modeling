import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipw
from functools import lru_cache
import inspect
from matplotlib.collections import LineCollection


# === 1. GENERIC CACHED ESR FUNCTION ===
def make_cached_spectra(esr_solver, so=None, sto=None, Bzvals=None, decimals_cache=4):
    """
    Factory that returns a cached ESR spectra function.
    
    Args:
        esr_solver: function like esrst.esr_spectra(ham, Bzvals)
        so, sto: optional spin operators
        Bzvals: np.array of B-field values
        decimals_cache: rounding precision for caching
        
    Returns:
        cached_spectra(ham_class, **params)
    """
    if Bzvals is None:
        raise ValueError("You must supply Bzvals to make_cached_spectra().")

    @lru_cache(maxsize=128)
    def cached_spectra(ham_class, **params):
        """Compute ESR spectra for any Hamiltonian class and any params."""
        sig = inspect.signature(ham_class)
        kwargs = {}

        if 'so' in sig.parameters and so is not None:
            kwargs['so'] = so
        if 'sto' in sig.parameters and sto is not None:
            kwargs['sto'] = sto

        # Add adjustable parameters (rounded for cache stability)
        kwargs.update({k: round(v, decimals_cache) for k, v in params.items()})

        ham = ham_class(**kwargs)
        ftran, eprob = esr_solver(ham, Bzvals)
        return ftran / 1e9, eprob  # convert to GHz

    return cached_spectra


# === 2. GENERIC UPDATER ===
def make_updater(ax, linecols, ham_class, cached_spectra, Bzvals, decimals_cache=4):
    """Return an update function that takes param sliders dynamically."""
    def update(**kwargs):
        params = {k: round(v, decimals_cache) for k, v in kwargs.items() if k != 'cscale'}
        cscale = kwargs.get('cscale', 1.0)
        
        ftran, eprob = cached_spectra(ham_class, **params)
        ntrans = ftran.shape[1]

        for i in range(ntrans):
            points = np.array([Bzvals, ftran[:, i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            linecols[i].set_segments(segments)

            prob = np.clip(eprob[:, i], 0, 1)
            alpha = np.clip(prob * cscale, 0.05, 1.0)
            colors = np.ones((len(alpha)-1, 4))
            colors[:, :3] = (0, 0, 0)  # black lines
            colors[:, 3] = alpha[:-1]
            linecols[i].set_colors(colors)

        ax.figure.canvas.draw_idle()
    return update


# === 3. INITIALIZATION ===
def interactive_spectrum(
    ham_class,
    param_ranges,
    param_defaults,
    ax,
    cached_spectra,
    Bzvals,
    cscale_range=(0, 1)
):
    """Build full interactive ESR viewer for any Hamiltonian class."""
    f0, e0 = cached_spectra(ham_class, **param_defaults)
    ntrans = f0.shape[1]

    # initial line collections
    linecols = []
    for i in range(ntrans):
        points = np.array([Bzvals, f0[:, i]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = np.ones((len(Bzvals)-1, 4))
        colors[:, :3] = (0, 0, 0)
        colors[:, 3] = np.clip(e0[:, i], 0, 1)[:-1]
        lc = LineCollection(segments, colors=colors, lw=1.5)
        ax.add_collection(lc)
        linecols.append(lc)

    layout = ipw.Layout(width='900px')
    sliders = {
        p: ipw.FloatSlider(
            min=v[0], max=v[1], step=v[2], value=param_defaults[p],
            description=p, layout=layout, continuous_update=False
        )
        for p, v in param_ranges.items()
    }
    sliders['cscale'] = ipw.FloatSlider(
        min=cscale_range[0], max=cscale_range[1], step=0.01,
        value=1.0, description='Color scale', layout=layout
    )

    update_func = make_updater(ax, linecols, ham_class, cached_spectra, Bzvals)
    out = ipw.interactive_output(update_func, sliders)
    display(ipw.VBox(list(sliders.values())), out)
