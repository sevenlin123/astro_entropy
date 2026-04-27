import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge

# ---------- Spectral model ----------
def gaussian_spectrum(wave, params):
    """
    params: array-like of shape (k, 3)
            each row = [amplitude, center, width]
    returns flux array
    """
    flux = np.zeros_like(wave, dtype=float)
    for amp, cen, wid in params:
        flux += amp * np.exp(-0.5 * ((wave - cen) / wid)**2)
    return flux

def generate_spectra(n_spec, wave, n_gauss=3):
    """
    Generate `n_spec` spectra by randomizing Gaussian parameters.
    """
    specs = []
    for _ in range(n_spec):
        amps   = np.random.uniform(0.5, 1.5, size=n_gauss)
        cents  = np.random.uniform(wave.min()+50, wave.max()-50, size=n_gauss)
        widths = np.random.uniform(20, 70, size=n_gauss)
        params = np.column_stack([amps, cents, widths])
        specs.append(gaussian_spectrum(wave, params))
    return np.array(specs)  # shape (n_spec, len(wave))

# ---------- Filter set ----------
def gaussian_filter(wave, center, width):
    """Transmittance curve of a Gaussian filter."""
    return norm.pdf(wave, loc=center, scale=width)

def make_filters(n_filt, wave):
    centers = np.linspace(wave.min()+30, wave.max()-30, n_filt)
    widths  = np.full(n_filt, 40.0)
    filt_mat = np.vstack([gaussian_filter(wave, c, w) for c,w in zip(centers,widths)])
    return filt_mat  # shape (n_filt, len(wave))

# ---------- Photometry ----------
def compute_photometry(spectra, filters, wave):
    """
    Integrate spectra through filters.
    spectra: (n_spec, len_wave)
    filters : (n_filt, len_wave)
    returns (n_spec, n_filt)
    """
    dw = wave[1] - wave[0]
    return (spectra @ filters.T) * dw

# ---------- Reconstruction ----------
def reconstruct_from_photometry(target_phot, basis_phot, basis_specs):
    """
    Fit linear combination of basis spectra to match target photometry.
    Returns reconstructed spectrum.
    """
    reg = Ridge(alpha=1e-3, fit_intercept=False)
    reg.fit(basis_phot.T, target_phot.T)   # basis_phot: (n_basis, n_filt)
    coeffs = reg.coef_.T                    # shape (n_basis,)
    recon_spec = coeffs @ basis_specs       # weighted sum of spectra
    return recon_spec

# ---------- Main demo ----------
if __name__ == "__main__":
    # Wavelength grid
    wave = np.arange(300, 2500 + 0.5, 0.5)   # nm

    # Generate spectra
    n_train = 80
    train_specs = generate_spectra(n_train, wave)

    # Filters
    n_filters = 6
    filt_mat = make_filters(n_filters, wave)

    # Photometry for training set
    train_phot = compute_photometry(train_specs, filt_mat, wave)

    # Target spectrum (randomly chosen from training set)
    idx = np.random.randint(n_train)
    target_spec = train_specs[idx]
    target_phot = train_phot[idx]

    # Reconstruct target spectrum
    recon_spec = reconstruct_from_photometry(target_phot, train_phot, train_specs)

    # Print comparison of fluxes (first 10 values)
    print("Target flux[0:10]:", target_spec[:10])
    print("Reconstructed flux[0:10]:", recon_spec[:10])