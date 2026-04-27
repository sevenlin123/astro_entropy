import numpy as np
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Spectral model
# ----------------------------------------------------------------------
def generate_spectrum(params, wave):
    """
    Simple spectral model:
        continuum : linear in wavelength
        three Gaussian absorption lines
    params: dict with keys 'slope', 'offset', 'depths' (tuple of 3)
    """
    slope = params['slope']
    offset = params['offset']
    depths = params['depths']

    continuum = slope * (wave - 5000.) + offset

    # Gaussian lines
    centers = np.array([4000., 5000., 6000.])
    widths = np.array([50., 80., 60.])

    line = np.zeros_like(wave)
    for d, c, w in zip(depths, centers, widths):
        line += -d * np.exp(-0.5 * ((wave - c) / w)**2)

    return continuum + line

# ----------------------------------------------------------------------
# Filter definitions
# ----------------------------------------------------------------------
def tophat_filter(center, width, wave):
    """Return a boolean mask for a top‑hat filter."""
    return (wave >= center - width/2.) & (wave <= center + width/2.)

def get_filters(wave):
    """Return a list of filter masks."""
    centers = [3600., 4400., 5500., 6500.]
    widths  = [100., 100., 100., 100.]
    return [tophat_filter(c, w, wave) for c, w in zip(centers, widths)]

# ----------------------------------------------------------------------
# Photometry calculation
# ----------------------------------------------------------------------
def photometry_from_spectrum(spectrum, wave, filters):
    """Compute integrated flux through each filter (simple sum)."""
    delta = np.mean(np.diff(wave))
    return np.array([np.sum(spectrum[f] * delta) for f in filters])

# ----------------------------------------------------------------------
# Library generation
# ----------------------------------------------------------------------
def random_params():
    """Generate random spectral parameters."""
    slope   = np.random.uniform(-0.02, 0.02)
    offset  = np.random.uniform(0.8, 1.2)
    depths  = tuple(np.random.uniform(0.05, 0.15, size=3))
    return {'slope': slope, 'offset': offset, 'depths': depths}

def build_library(n, wave):
    """Return an array of spectra forming a library."""
    lib = []
    for _ in range(n):
        params = random_params()
        lib.append(generate_spectrum(params, wave))
    return np.vstack(lib)   # shape (n, len(wave))

# ----------------------------------------------------------------------
# Reconstruction
# ----------------------------------------------------------------------
def reconstruct_spectrum(phot, filters, wave, lib_spectra):
    """
    Solve for linear coefficients that reproduce the photometry,
    then build the reconstructed spectrum.
    """
    n_filters = len(filters)
    n_lib     = lib_spectra.shape[0]
    delta = np.mean(np.diff(wave))

    # Build design matrix A where A[j,i] = integral of lib_spectra[i] over filter j
    A = np.zeros((n_filters, n_lib))
    for j, f in enumerate(filters):
        for i, spec in enumerate(lib_spectra):
            A[j, i] = np.sum(spec[f] * delta)

    # Least‑squares solution
    coeffs = np.linalg.lstsq(A, phot, rcond=None)[0]

    # Reconstruct spectrum
    recon = np.dot(coeffs, lib_spectra)
    return recon, coeffs

# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Wavelength grid
    wave = np.linspace(3500., 7500., 4000)

    # True spectrum
    true_params = {'slope': 0.005, 'offset': 1.0, 'depths': (0.08, 0.12, 0.10)}
    true_spec   = generate_spectrum(true_params, wave)

    # Filters
    filt_masks = get_filters(wave)

    # Synthetic photometry
    phot_true = photometry_from_spectrum(true_spec, wave, filt_masks)

    # Build library
    library = build_library(30, wave)

    # Reconstruct
    recon_spec, coeffs = reconstruct_spectrum(phot_true, filt_masks, wave, library)

    # Output simple diagnostics
    print("True photometry:", phot_true)
    print("Reconstructed photometry:", photometry_from_spectrum(recon_spec, wave, filt_masks))
    print("Mean absolute error on spectrum:", np.mean(np.abs(recon_spec - true_spec)))
    print("Coefficients shape:", coeffs.shape)