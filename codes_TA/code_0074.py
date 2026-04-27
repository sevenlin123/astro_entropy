import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_basis(num_wavelengths, num_components, seed=None):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(num_wavelengths, num_components))
    Q, _ = np.linalg.qr(A)  # orthonormalize
    return Q

def generate_synthetic_spectra(num_spectra, num_wavelengths, num_components,
                               basis, noise_level=0.01, seed=None):
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(size=(num_spectra, num_components))
    spectra = coeffs @ basis.T
    spectra += rng.normal(scale=noise_level, size=spectra.shape)
    return spectra, coeffs

def create_filters(num_filters, num_wavelengths, seed=None):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, num_wavelengths)
    filters = np.zeros((num_filters, num_wavelengths))
    for i in range(num_filters):
        center = rng.uniform(0, 1)
        width = rng.uniform(0.05, 0.15)
        filters[i] = np.exp(-0.5 * ((x - center) / width)**2)
        filters[i] /= filters[i].sum()  # normalize
    return filters

def photometry_from_spectra(spectra, filters):
    return spectra @ filters.T  # (n_spectra, n_filters)

def reconstruct_spectra(photometry, filters, basis, alpha=1.0):
    # Train ridge regression to map photometry to coefficients
    ridge = Ridge(alpha=alpha)
    ridge.fit(photometry, ridge.coef_)  # placeholder to satisfy API
    # Instead learn mapping directly from photometry to coefficients
    ridge.fit(photometry, ridge.coef_)
    coeff_pred = ridge.predict(photometry)
    recon = coeff_pred @ basis.T
    return recon

def main():
    # Settings
    n_wl = 200          # number of wavelength points
    n_comp = 10         # number of spectral basis components
    n_spec = 500        # number of synthetic spectra
    n_filt = 6          # number of photometric filters
    seed = 42

    # Generate basis
    basis = create_basis(n_wl, n_comp, seed)

    # Generate synthetic spectra and coefficients
    spectra, coeffs_true = generate_synthetic_spectra(
        n_spec, n_wl, n_comp, basis, noise_level=0.02, seed=seed
    )

    # Create filter set
    filters = create_filters(n_filt, n_wl, seed+1)

    # Generate photometry
    phot = photometry_from_spectra(spectra, filters)

    # Split into training and testing
    idx_train, idx_test = train_test_split(np.arange(n_spec), test_size=0.2, random_state=seed)
    phot_train, phot_test = phot[idx_train], phot[idx_test]
    coeff_train, coeff_test = coeffs_true[idx_train], coeffs_true[idx_test]

    # Train Ridge regression to map photometry to coefficients
    reg = Ridge(alpha=1.0)
    reg.fit(phot_train, coeff_train)

    # Predict coefficients for test set
    coeff_pred = reg.predict(phot_test)

    # Reconstruct spectra
    spectra_rec = coeff_pred @ basis.T

    # Evaluate reconstruction error
    rmse = np.sqrt(mean_squared_error(spectra[idx_test], spectra_rec))
    print(f"Reconstruction RMSE over test spectra: {rmse:.4f}")

if __name__ == "__main__":
    main()