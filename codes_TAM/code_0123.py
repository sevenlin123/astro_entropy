import numpy as np
from scipy.integrate import simps
from sklearn.linear_model import LinearRegression

# ----- Define the spectral model -------------------------------------------------
def build_wavelength_grid(start=4000, stop=8000, num=1000):
    """Wavelength grid in Angstroms."""
    return np.linspace(start, stop, num)

def build_basis_functions(wavelength, n_basis=5):
    """Generate simple Gaussian basis spectra."""
    centers = np.linspace(4100, 7900, n_basis)
    widths  = 200 * np.ones_like(centers)
    basis   = []
    for c, w in zip(centers, widths):
        gauss = np.exp(-0.5 * ((wavelength - c)/w)**2)
        basis.append(gauss)
    return np.array(basis)            # shape (n_basis, len(wavelength))

def generate_synthetic_spectrum(coeffs, basis, noise_std=0.02):
    """
    coeffs: array of shape (n_samples, n_basis)
    basis : array of shape (n_basis, n_pixels)
    Returns: spectra array of shape (n_samples, n_pixels)
    """
    spectra = coeffs @ basis          # linear combination
    spectra += np.random.normal(0, noise_std, spectra.shape)
    return spectra

# ----- Define photometric filters -----------------------------------------------
def build_filters(wavelength, n_filters=3):
    """Gaussian filter transmission curves."""
    centers = np.linspace(4600, 6600, n_filters)
    widths  = 300 * np.ones_like(centers)
    filters = []
    for c, w in zip(centers, widths):
        trans = np.exp(-0.5 * ((wavelength - c)/w)**2)
        trans /= simps(trans, wavelength)   # normalize
        filters.append(trans)
    return np.array(filters)                 # shape (n_filters, n_pixels)

def compute_photometry(spectra, filters, wavelength):
    """
    spectra : (n_samples, n_pixels)
    filters : (n_filters, n_pixels)
    Returns: photometry array of shape (n_samples, n_filters)
    """
    n_samples = spectra.shape[0]
    n_filters = filters.shape[0]
    phot = np.zeros((n_samples, n_filters))
    for i in range(n_filters):
        phot[:, i] = simps(spectra * filters[i], wavelength, axis=1)
    return phot

# ----- Training and reconstruction ---------------------------------------------
def train_regressors(X, Y):
    """
    X: photometry (n_samples, n_filters)
    Y: coefficients (n_samples, n_basis)
    Returns list of trained regressors (one per basis coefficient)
    """
    regressors = []
    for j in range(Y.shape[1]):
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, Y[:, j])
        regressors.append(reg)
    return regressors

def reconstruct_spectrum_from_photometry(phot, regressors, basis):
    """
    phot: (n_samples, n_filters)
    regressors: list of trained regressors
    basis: (n_basis, n_pixels)
    Returns reconstructed spectra (n_samples, n_pixels)
    """
    coeffs_pred = np.column_stack([reg.predict(phot) for reg in regressors])
    return coeffs_pred @ basis

# ----- Main routine -------------------------------------------------------------
def main():
    np.random.seed(42)

    # Wavelength grid
    wav = build_wavelength_grid()
    n_pixels = wav.size

    # Basis spectra
    basis = build_basis_functions(wav)            # (n_basis, n_pixels)
    n_basis = basis.shape[0]

    # Filters
    filters = build_filters(wav)                  # (n_filters, n_pixels)
    n_filters = filters.shape[0]

    # Generate training data
    n_train = 1000
    coeffs_train = np.random.uniform(0.0, 1.0, size=(n_train, n_basis))
    spectra_train = generate_synthetic_spectrum(coeffs_train, basis)
    phot_train = compute_photometry(spectra_train, filters, wav)

    # Train regressors
    regressors = train_regressors(phot_train, coeffs_train)

    # Test data
    n_test = 200
    coeffs_test = np.random.uniform(0.0, 1.0, size=(n_test, n_basis))
    spectra_test = generate_synthetic_spectrum(coeffs_test, basis)
    phot_test = compute_photometry(spectra_test, filters, wav)

    # Reconstruct spectra from photometry
    spectra_rec = reconstruct_spectrum_from_photometry(phot_test, regressors, basis)

    # Evaluation
    mse = np.mean((spectra_test - spectra_rec)**2)
    print(f"Mean Squared Error on test set: {mse:.6f}")

if __name__ == "__main__":
    main()