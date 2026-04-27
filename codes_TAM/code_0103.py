import numpy as np
from sklearn.linear_model import Ridge
from scipy.signal import gaussian

# --------------------------------------------------------------------------- #
# 1. Spectral model definition – a set of Gaussian basis functions
# --------------------------------------------------------------------------- #
def build_basis(lam, n_bases=10, rng=np.random.default_rng(42)):
    """
    Build a set of Gaussian basis functions over wavelength grid `lam`.
    Returns a matrix of shape (len(lam), n_bases).
    """
    centers = np.linspace(lam[0], lam[-1], n_bases)
    widths = np.full(n_bases, (lam[-1]-lam[0])/n_bases/2)   # fixed width
    basis = np.empty((len(lam), n_bases))
    for i, (c, w) in enumerate(zip(centers, widths)):
        basis[:, i] = np.exp(-0.5 * ((lam - c)/w)**2)
    return basis


# --------------------------------------------------------------------------- #
# 2. Synthetic spectra generation
# --------------------------------------------------------------------------- #
def generate_spectra(n_samples, basis, rng=np.random.default_rng(123)):
    """
    Generate synthetic spectra as random linear combinations of `basis`.
    """
    coeffs = rng.normal(size=(n_samples, basis.shape[1]))
    spectra = coeffs @ basis.T  # shape (n_samples, len(lam))
    return spectra, coeffs


# --------------------------------------------------------------------------- #
# 3. Photometric data generation
# --------------------------------------------------------------------------- #
def build_filters(lam, n_filters=3):
    """
    Construct simple Gaussian filter transmission curves.
    """
    filters = []
    centers = np.linspace(lam[0]+50, lam[-1]-50, n_filters)
    width = (lam[-1]-lam[0]) / 20.0
    for c in centers:
        filt = np.exp(-0.5 * ((lam - c)/width)**2)
        filt /= filt.max()          # normalise peak to 1
        filters.append(filt)
    return np.array(filters)        # shape (n_filters, len(lam))


def compute_photometry(spectra, filters, lam):
    """
    Integrate spectra through each filter to obtain photometric fluxes.
    Simple Riemann sum approximation.
    """
    # normalise filter integrals to avoid scaling differences
    filt_norm = filters / np.trapz(filters, lam, axis=1)[:, None]
    # dot product gives integrated flux per filter
    return spectra @ filt_norm.T    # shape (n_samples, n_filters)


# --------------------------------------------------------------------------- #
# 4. Reconstruction framework
# --------------------------------------------------------------------------- #
def train_reconstructor(photometry, spectra, alpha=1.0):
    """
    Train a linear model that maps photometric fluxes to spectra.
    """
    reg = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    reg.fit(photometry, spectra)
    return reg


def reconstruct_spectrum(regressor, photometry):
    """
    Predict a spectrum given photometric fluxes.
    """
    return regressor.predict(photometry)


# --------------------------------------------------------------------------- #
# 5. Example workflow
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(2024)

    # Wavelength grid (400–900 nm, 500 points)
    lam = np.linspace(400, 900, 500)

    # Build basis and generate training data
    basis = build_basis(lam, n_bases=12, rng=rng)
    spectra_train, _ = generate_spectra(300, basis, rng=rng)

    # Build filters and compute photometry for training
    filters = build_filters(lam, n_filters=4)
    phot_train = compute_photometry(spectra_train, filters, lam)

    # Train reconstructor
    reconstructor = train_reconstructor(phot_train, spectra_train, alpha=0.1)

    # Generate test spectra and their photometry
    spectra_test, coeffs_test = generate_spectra(50, basis, rng=rng)
    phot_test = compute_photometry(spectra_test, filters, lam)

    # Reconstruct spectra from photometry
    spectra_rec = reconstruct_spectrum(reconstructor, phot_test)

    # Evaluate reconstruction error (mean absolute error)
    mae = np.mean(np.abs(spectra_test - spectra_rec))
    print(f"Mean absolute error of reconstructed spectra: {mae:.4f}")

    # Optional: display a single example (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        idx = 0
        plt.figure(figsize=(8, 4))
        plt.plot(lam, spectra_test[idx], label='True spectrum')
        plt.plot(lam, spectra_rec[idx], '--', label='Reconstructed')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Flux (arb. units)')
        plt.title('Spectral reconstruction example')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass