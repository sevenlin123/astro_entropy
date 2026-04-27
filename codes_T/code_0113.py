import numpy as np
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Spectral model: sum of Gaussian lines
# ----------------------------------------------------------------------
def gaussian(wl, amp, cen, wid):
    """Gaussian line."""
    return amp * np.exp(-0.5 * ((wl - cen) / wid) ** 2)

def synthetic_spectrum(wl, params):
    """
    Build a synthetic spectrum as a sum of Gaussian lines.
    
    Parameters
    ----------
    wl : ndarray
        Wavelength grid.
    params : list of dict
        Each dict has keys 'amp', 'cen', 'wid'.
        
    Returns
    -------
    spec : ndarray
        Spectrum values at wavelengths `wl`.
    """
    spec = np.zeros_like(wl)
    for p in params:
        spec += gaussian(wl, p['amp'], p['cen'], p['wid'])
    return spec

# ----------------------------------------------------------------------
# Filters: simple Gaussian bandpasses
# ----------------------------------------------------------------------
def gaussian_filter(wl, cen, wid):
    """Filter transmission curve."""
    return np.exp(-0.5 * ((wl - cen) / wid) ** 2)

def build_filters():
    """
    Create a set of filter response curves.
    
    Returns
    -------
    filters : list of tuples
        Each tuple is (name, transmission_function)
    """
    wl = np.linspace(3000, 8000, 5000)  # Angstrom
    filt_specs = [
        ('U', 3500, 400),
        ('B', 4400, 350),
        ('V', 5500, 300),
        ('R', 6500, 250),
        ('I', 7800, 200),
    ]
    filters = []
    for name, cen, wid in filt_specs:
        trans = gaussian_filter(wl, cen, wid)
        filters.append((name, trans))
    return wl, filters

# ----------------------------------------------------------------------
# Photometry calculation
# ----------------------------------------------------------------------
def compute_photometry(spec, wl, filters):
    """
    Compute synthetic photometric fluxes by integrating spectrum times filter.
    
    Parameters
    ----------
    spec : ndarray
        Spectrum values at wavelengths `wl`.
    wl : ndarray
        Wavelength grid.
    filters : list of tuples
        Each tuple is (name, transmission_curve).
        
    Returns
    -------
    phot : dict
        Photometric fluxes keyed by filter name.
    """
    phot = {}
    for name, trans in filters:
        flux = np.trapz(spec * trans, wl) / np.trapz(trans, wl)
        phot[name] = flux
    return phot

# ----------------------------------------------------------------------
# Reconstruction from photometry
# ----------------------------------------------------------------------
def build_design_matrix(filters, wl, line_params):
    """
    Build design matrix relating line amplitudes to photometric fluxes.
    
    Parameters
    ----------
    filters : list of tuples
        (name, transmission_curve).
    wl : ndarray
        Wavelength grid.
    line_params : list of dict
        Parameters of basis Gaussian lines (without amplitudes).
        
    Returns
    -------
    A : ndarray
        Design matrix of shape (n_filters, n_lines).
    """
    n_filters = len(filters)
    n_lines = len(line_params)
    A = np.zeros((n_filters, n_lines))
    for i, (_, trans) in enumerate(filters):
        for j, p in enumerate(line_params):
            g = gaussian(wl, 1.0, p['cen'], p['wid'])
            A[i, j] = np.trapz(g * trans, wl) / np.trapz(trans, wl)
    return A

def reconstruct_amplitudes(photometry, A):
    """
    Reconstruct line amplitudes from photometry using ridge regression.
    
    Parameters
    ----------
    photometry : ndarray
        Array of photometric fluxes (ordered as in A).
    A : ndarray
        Design matrix.
        
    Returns
    -------
    amps : ndarray
        Recovered amplitudes.
    """
    reg = Ridge(alpha=1.0, fit_intercept=False)
    reg.fit(A, photometry)
    return reg.coef_

def reconstruct_spectrum(wl, line_params, amps):
    """
    Reconstruct full spectrum from recovered amplitudes.
    
    Parameters
    ----------
    wl : ndarray
        Wavelength grid.
    line_params : list of dict
        Line parameters without amplitudes.
    amps : ndarray
        Recovered amplitudes.
        
    Returns
    -------
    spec_rec : ndarray
        Reconstructed spectrum.
    """
    spec_rec = np.zeros_like(wl)
    for amp, p in zip(amps, line_params):
        spec_rec += gaussian(wl, amp, p['cen'], p['wid'])
    return spec_rec

# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)

    # 1. Define wavelength grid
    wl = np.linspace(3000, 8000, 5000)  # Angstrom

    # 2. Define true line parameters (amplitude, center, width)
    true_params = [
        {'amp': 1.0,  'cen': 3600, 'wid': 30},
        {'amp': 0.8,  'cen': 4700, 'wid': 45},
        {'amp': 0.6,  'cen': 5800, 'wid': 35},
        {'amp': 0.4,  'cen': 6900, 'wid': 25},
        {'amp': 0.2,  'cen': 8100, 'wid': 20},
    ]

    # 3. Generate synthetic spectrum
    true_spec = synthetic_spectrum(wl, true_params)

    # 4. Build filters
    wl_filt, filters = build_filters()

    # 5. Compute synthetic photometry
    phot_true = compute_photometry(true_spec, wl_filt, filters)
    phot_array = np.array([phot_true[name] for name, _ in filters])

    # 6. Prepare basis line parameters (same centers and widths, unknown amplitudes)
    basis_params = [{'cen': p['cen'], 'wid': p['wid']} for p in true_params]

    # 7. Build design matrix
    A = build_design_matrix(filters, wl_filt, basis_params)

    # 8. Reconstruct amplitudes
    recovered_amps = reconstruct_amplitudes(phot_array, A)

    # 9. Reconstruct spectrum
    rec_spec = reconstruct_spectrum(wl, basis_params, recovered_amps)

    # 10. Evaluate
    rms_error = np.sqrt(np.mean((true_spec - rec_spec) ** 2))
    print(f"Reconstruction RMS error: {rms_error:.4f}")

    # Print true vs recovered amplitudes
    print("\nTrue vs Recovered Amplitudes:")
    for i, (t, r) in enumerate(zip([p['amp'] for p in true_params], recovered_amps)):
        print(f"Line {i+1}: true={t:.3f}, recovered={r:.3f}")