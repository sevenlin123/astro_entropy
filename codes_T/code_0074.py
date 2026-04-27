import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# ----- Spectral model ---------------------------------------------------------

def gaussian_basis(wave, centers, sigma=20):
    """Return a matrix of Gaussian basis functions."""
    G = np.exp(-0.5 * ((wave[:, None] - centers[None, :]) / sigma)**2)
    return G

def create_spectral_model():
    wave = np.linspace(300, 800, 501)          # nm
    centers = np.arange(350, 750, 50)         # nm
    basis = gaussian_basis(wave, centers)
    return wave, basis

# ----- Synthetic spectra -----------------------------------------------------

def generate_synthetic_spectra(n_spectra, basis, rng=None):
    rng = np.random.default_rng(rng)
    coeffs = rng.standard_normal((n_spectra, basis.shape[1]))
    spectra = coeffs @ basis.T                 # shape (n_spectra, len(wave))
    spectra += rng.normal(scale=0.05, size=spectra.shape)  # noise
    return spectra, coeffs

# ----- Photometric system ----------------------------------------------------

def create_filters():
    """Return list of simple boxcar filter transmission curves."""
    wave = np.linspace(300, 800, 501)
    filt_edges = [(350, 450), (460, 560), (570, 670), (680, 780)]
    filters = []
    for lo, hi in filt_edges:
        trans = np.where((wave >= lo) & (wave <= hi), 1.0, 0.0)
        filters.append(trans)
    return wave, filters

def compute_photometry(spectra, wave, filters):
    """Integrate spectra over filters."""
    fluxes = []
    for filt in filters:
        flux = np.trapz(spectra * filt, wave, axis=1)
        fluxes.append(flux)
    return np.vstack(fluxes).T   # shape (n_spectra, n_filters)

# ----- Reconstruction --------------------------------------------------------

def reconstruct_from_photometry(photon_data, spectra_train, X_train, X_test):
    """
    Fit ridge regression to predict spectra from photometry.
    Returns predictions for X_test.
    """
    reg = Ridge(alpha=1.0, max_iter=2000)
    reg.fit(X_train, spectra_train)
    preds = reg.predict(X_test)
    return preds

# ----- Main routine ----------------------------------------------------------

def main():
    rng_seed = 42

    # Build spectral model
    wave, basis = create_spectral_model()

    # Generate synthetic spectra
    n_spectra = 200
    spectra, coeffs = generate_synthetic_spectra(n_spectra, basis, rng=rng_seed)

    # Create photometric system
    _, filters = create_filters()

    # Compute photometric measurements
    photometry = compute_photometry(spectra, wave, filters)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        photometry, spectra, test_size=0.2, random_state=rng_seed
    )

    # Reconstruct spectra from photometry
    spectra_pred = reconstruct_from_photometry(photometry, y_train, X_train, X_test)

    # Simple evaluation
    mse = np.mean((spectra_test - spectra_pred)**2)
    print(f"Mean squared error of reconstruction: {mse:.6f}")

if __name__ == "__main__":
    main()