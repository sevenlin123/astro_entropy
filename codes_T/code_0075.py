import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# ---------- Spectral model ----------
def gaussian_basis(wl, center, width):
    """One Gaussian basis function."""
    return np.exp(-0.5 * ((wl - center) / width) ** 2)

def create_bases(wl, n_bases=5):
    """Generate a list of Gaussian basis functions."""
    np.random.seed(0)
    centers = np.linspace(wl.min(), wl.max(), n_bases)
    widths = np.full(n_bases, (wl.max() - wl.min()) / (3 * n_bases))
    bases = [gaussian_basis(wl, c, w) for c, w in zip(centers, widths)]
    return np.vstack(bases).T   # shape (n_wl, n_bases)

def generate_synthetic_spectra(n_samples, wl, bases):
    """Sample spectra as random linear combinations of basis functions."""
    rng = np.random.default_rng(seed=1)
    coeffs = rng.normal(size=(n_samples, bases.shape[1]))
    spectra = coeffs @ bases.T   # shape (n_samples, n_wl)
    return spectra, coeffs

# ---------- Photometric model ----------
def create_filters(wl):
    """Simple top‑hat filters for U, B, V, R, I bands."""
    filters = []
    bands = [(350, 400), (440, 500), (530, 590), (630, 690), (760, 820)]  # nm
    for low, high in bands:
        filt = np.where((wl >= low) & (wl <= high), 1.0, 0.0)
        filters.append(filt)
    return np.vstack(filters).T  # shape (n_wl, n_filters)

def generate_photometry(spectra, filters):
    """Compute photometric fluxes by integrating spectra with filter curves."""
    # Integral approximated by dot product over wavelength axis
    return spectra @ filters  # shape (n_samples, n_filters)

# ---------- Reconstruction ----------
def reconstruct_spectra(photometry, wl, bases, alpha=1.0):
    """
    Reconstruct spectra from photometry using Ridge regression
    (multi‑output regression: one output per wavelength point).
    """
    n_filters = photometry.shape[1]
    n_wl = wl.size
    # Fit ridge regression for each wavelength point
    model = Ridge(alpha=alpha, fit_intercept=False, solver='auto')
    model.fit(photometry, bases)  # bases are our "true" spectra
    reconstructed = model.predict(photometry)
    return reconstructed

# ---------- Example workflow ----------
if __name__ == "__main__":
    # Wavelength grid
    wl = np.arange(300, 901, 1)          # 300–900 nm, 1 nm steps

    # Generate basis functions and spectra
    bases = create_bases(wl, n_bases=10)
    spectra, coeffs = generate_synthetic_spectra(200, wl, bases)

    # Create filters and compute photometry
    filters = create_filters(wl)
    photometry = generate_photometry(spectra, filters)

    # Split into training and testing sets
    split = int(0.8 * spectra.shape[0])
    X_train, X_test = photometry[:split], photometry[split:]
    y_train, y_test = spectra[:split], spectra[split:]

    # Train reconstruction model
    reconstructor = Ridge(alpha=1.0, fit_intercept=False)
    reconstructor.fit(X_train, y_train)

    # Predict spectra for test set
    y_pred = reconstructor.predict(X_test)

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean squared error on test set: {mse:.4f}")

    # Visual comparison for a single sample (optional)
    idx = 0
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,4))
    plt.plot(wl, y_test[idx], label="True spectrum")
    plt.plot(wl, y_pred[idx], label="Reconstructed spectrum", linestyle="--")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux (arbitrary units)")
    plt.title("Spectral reconstruction from photometry")
    plt.legend()
    plt.tight_layout()
    plt.show()