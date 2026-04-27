import numpy as np
from sklearn.linear_model import Ridge

# ----------------------------------------------------------------------
# 1. Define wavelength grid and basis spectra
# ----------------------------------------------------------------------
def make_wave_grid(n=1000, wl_min=400.0, wl_max=800.0):
    return np.linspace(wl_min, wl_max, n)

def gaussian_spectrum(wl, center, width):
    return np.exp(-0.5 * ((wl - center) / width) ** 2)

def build_basis(wl):
    # Three simple Gaussian basis components
    b1 = gaussian_spectrum(wl, 450.0, 30.0)
    b2 = gaussian_spectrum(wl, 550.0, 40.0)
    b3 = gaussian_spectrum(wl, 650.0, 35.0)
    return np.vstack([b1, b2, b3])  # shape (3, len(wl))

# ----------------------------------------------------------------------
# 2. Generate synthetic spectra as random linear combos of basis
# ----------------------------------------------------------------------
def generate_synthetic_spectra(basis, nsamples=200, noise_std=0.02):
    n_basis, n_wl = basis.shape
    coeffs = np.random.rand(nsamples, n_basis)          # random positive weights
    spectra = coeffs @ basis                           # linear combination
    spectra += noise_std * np.random.randn(*spectra.shape)
    return spectra, coeffs

# ----------------------------------------------------------------------
# 3. Define photometric filters and compute photometry
# ----------------------------------------------------------------------
def build_filters(wl, n_filters=5):
    # Simple Gaussian filter responses
    centers = np.linspace(420.0, 720.0, n_filters)
    widths = np.full(n_filters, 30.0)
    filters = [gaussian_spectrum(wl, c, w) for c, w in zip(centers, widths)]
    return np.array(filters)   # shape (n_filters, len(wl))

def compute_photometry(spectra, wl, filters):
    # Integrate (flux * filter) over wavelength
    phots = []
    for filt in filters:
        # trapezoidal integration
        flux_filt = spectra * filt[np.newaxis, :]      # broadcast over samples
        phot = np.trapz(flux_filt, wl, axis=1)         # integrate each sample
        phots.append(phot)
    return np.stack(phots, axis=1)                     # shape (nsamples, n_filters)

# ----------------------------------------------------------------------
# 4. Reconstruct spectra from photometry using ridge regression
# ----------------------------------------------------------------------
def train_reconstruction_model(photometry, coeffs, alpha=1.0):
    # Fit a linear model: coeffs = W * photometry + bias
    reg = Ridge(alpha=alpha, fit_intercept=True, normalize=True)
    reg.fit(photometry, coeffs)
    return reg

def reconstruct_spectra(model, photometry, basis):
    # Predict coefficients from photometry
    pred_coeffs = model.predict(photometry)             # shape (nsamples, n_basis)
    # Reconstruct spectra
    return pred_coeffs @ basis.T                        # shape (nsamples, len(wl))

# ----------------------------------------------------------------------
# 5. Main routine
# ----------------------------------------------------------------------
def main():
    # Setup
    wl = make_wave_grid()
    basis = build_basis(wl)
    filters = build_filters(wl)

    # Generate synthetic data
    spectra, true_coeffs = generate_synthetic_spectra(basis)
    photometry = compute_photometry(spectra, wl, filters)

    # Split into train/test
    n_train = int(0.8 * spectra.shape[0])
    X_train, X_test = photometry[:n_train], photometry[n_train:]
    y_train, y_test = true_coeffs[:n_train], true_coeffs[n_train:]

    # Train regression model
    model = train_reconstruction_model(X_train, y_train)

    # Reconstruct on test set
    recon_test = reconstruct_spectra(model, X_test, basis)

    # Evaluate: mean absolute error per wavelength
    mae = np.mean(np.abs(recon_test - spectra[n_train:]), axis=0)
    print("Mean Absolute Error per wavelength:", mae)
    print("Overall MAE:", np.mean(mae))

if __name__ == "__main__":
    main()