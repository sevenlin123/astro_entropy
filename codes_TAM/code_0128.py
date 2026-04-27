import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# Define wavelength grid (in nm)
WAVELENGTHS = np.linspace(300, 900, 601)  # 300–900 nm

# ----------------------------------------------------------------------
def blackbody_lambda(wl, temp):
    """Planck function in units of W sr-1 m-2 nm-1."""
    wl_m = wl * 1e-9
    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23
    intensity = (2.0*h*c**2) / (wl_m**5) / (np.exp(h*c/(wl_m*k*temp)) - 1.0)
    return intensity

def gaussian_line(wl, center, sigma, amplitude):
    """Gaussian emission/absorption line."""
    return amplitude * np.exp(-0.5 * ((wl - center)/sigma)**2)

def spectrum_model(wl, temp, n_lines=3):
    """Synthetic spectrum: blackbody + random Gaussian lines."""
    spec = blackbody_lambda(wl, temp)
    rng = np.random.default_rng()
    for _ in range(n_lines):
        center = rng.uniform(350, 850)
        sigma  = rng.uniform(5, 20)
        amp    = rng.uniform(-0.5, 0.5)  # negative for absorption
        spec += gaussian_line(wl, center, sigma, amp)
    return spec

# ----------------------------------------------------------------------
def gaussian_filter_response(wl, center, width):
    """Filter transmission curve as a Gaussian."""
    return np.exp(-0.5 * ((wl - center)/width)**2)

# Predefined filters (center nm, width nm)
FILTERS = {
    'U': (365, 40),
    'B': (445, 45),
    'V': (551, 50),
}

def apply_filter(spectrum, wl, filt_center, filt_width):
    """Integrate spectrum over a filter transmission."""
    trans = gaussian_filter_response(wl, filt_center, filt_width)
    return simps(spectrum * trans, wl) / simps(trans, wl)

def generate_photometry(spectra, wl):
    """Compute synthetic magnitudes (fluxes) for all spectra."""
    n_spec = spectra.shape[0]
    phot = np.empty((n_spec, len(FILTERS)))
    for i, filt in enumerate(FILTERS.values()):
        phot[:, i] = np.array([apply_filter(spec, wl, *filt) for spec in spectra])
    return phot

# ----------------------------------------------------------------------
def reconstruct_spectrum(photometry, wl, n_basis=10):
    """
    Reconstruct spectrum as linear combination of basis functions.
    Here we use polynomial basis (order up to n_basis-1).
    """
    # Build design matrix for basis functions evaluated at each wavelength
    X_basis = np.vstack([wl**k for k in range(n_basis)]).T  # shape (N_wl, n_basis)

    # Train ridge regression on each wavelength coefficient
    coeffs = []
    for j in range(len(wl)):
        y = spectra_train[:, j]  # true fluxes at wavelength j
        ridge = Ridge(alpha=1.0)
        ridge.fit(photometry_train, y)
        coeffs.append(ridge.coef_)
    coeffs = np.array(coeffs).T  # shape (len(filters), n_wl)

    # Predict coefficients for new photometry
    pred_coeffs = photometry @ coeffs.T  # shape (n_new, n_wl)

    # Reconstruct spectra
    reconstructed = pred_coeffs  # each row is the predicted spectrum
    return reconstructed

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Generate synthetic training data
    rng = np.random.default_rng(42)
    n_train = 200
    temps = rng.uniform(4000, 12000, size=n_train)  # temperatures in K
    spectra_train = np.array([spectrum_model(WAVELENGTHS, t) for t in temps])

    # Generate synthetic photometry
    photometry_train = generate_photometry(spectra_train, WAVELENGTHS)

    # Generate a test spectrum
    temp_test = 7500
    spectrum_test = spectrum_model(WAVELENGTHS, temp_test)
    photometry_test = generate_photometry(np.array([spectrum_test]), WAVELENGTHS)

    # Reconstruct spectrum from photometry
    # Simple linear regression per wavelength (ridge)
    ridge = Ridge(alpha=1.0)
    ridge.fit(photometry_train, spectra_train)
    spectrum_pred = ridge.predict(photometry_test)[0]

    # Print results
    print(f"True temperature: {temp_test:.0f} K")
    print(f"Reconstructed spectrum shape: {spectrum_pred.shape}")
    print(f"First 10 values of true vs predicted:")
    for i in range(10):
        print(f"  w={WAVELENGTHS[i]:4.0f} nm  true={spectrum_test[i]:.3e}  pred={spectrum_pred[i]:.3e}")